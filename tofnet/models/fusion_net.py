import warnings
import itertools

from tofnet.models.blocks import create_cba, get_all_blocks, SEModule
from tofnet.models.model_wrapper import ModelWrapper
from tofnet.models.unet import SubIdentity, ConvBlock, UpSampling

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(
        self, model_cfg, in_width, n_inputs, Block=None, width=None,
        ksize=None, bn=True, merge="add", multi_inputs=None
    ):
        super().__init__()
        assert merge in {"concat", "add"}, "Only add and concat are supported"
        self.concatenate = (merge=="concat")
        if self.concatenate:
            assert None not in [width, Block, ksize],\
                "Block, width and ksize should be set when concatenating"

        if multi_inputs is not None:
            self.multi_inputs_fusion = Block(
                np.sum(multi_inputs), in_width, 1, bn=bn, act=True
            )
            n_inputs += 1

        self.n_inputs = n_inputs
        self.se = nn.ModuleList([
            SEModule(model_cfg, in_width, ratio=4) for _ in range(n_inputs)
        ])

        if self.concatenate:
            assert multi_inputs != None, "Concat and multi_inputs is not supported"
            self.fusion = Block(
                model_cfg, in_width*n_inputs,
                width, ksize, bn=bn, act=True
            )

    def forward(self, nx):
        if hasattr(self, "multi_inputs_fusion"):
            _in = nx
            nx = list(nx[:self.n_inputs-1])
            nx += [self.multi_inputs_fusion(torch.cat(_in[self.n_inputs-1:], dim=1))]

        nx = [se(xx) for se, xx in zip(self.se, nx)]
        if self.concatenate:
            nx = torch.cat(nx, dim=1)
        else:
            nx = torch.stack(nx).sum(dim=0)
        if self.concatenate:
            nx = self.fusion(nx)
        return nx


class FusionNet(ModelWrapper):
    """ A Configurable Unet model, with slow fusion

    Arguments:
        block (str): name of the conv-block (enum: double, residual etc.),
                    :mod:`blocks` for more options
        last_act (str): the final activation function
        conv_transpose (bool): wether to use a conv-transpose (True) or upsampling-conv (False)
        bn (bool): wether to use batch normalization or not
        small_drop (float): dropout ratio at the higher levels of the network
        big_drop (float): dropout ratio at the lower levels of the network, should be >small_drop
        sddrop (float): ratio of stochastic depth dropout (only available with residual or mobile blocks)
        se_ratio (float): squeeze-and-excite ratio
        input_format (list): list of strings with the input types
        output_format (list): list of strings with the output types
        multi_input (bool): wether to use multi-level inputs
        multi_scale (bool): wether to use multi-level outputs

    .. note::

        The model_cfg, input_size and classes arguments are set inside the code
        based on other config information and should not be set manually
    """
    def __init__(self, model_cfg,block='residual',
            input_size=(1,256,256), classes=None, last_act="linear",
            conv_transpose=False, bn=True, architecture=None,
            big_drop=0., small_drop=0., sddrop=0., se_ratio=0.0,
            input_format=None, output_format=None, multi_scale=False,
            multi_input=False):
        # Parse the network's architecture
        if architecture is None:
            architecture = {
                "first": 32,
                "enc": {
                    "width": [16, 32, 48, 96],
                    "repeat": [2, 3, 3, 4]
                },
                "dec": {
                    "width": [48, 32, 32],
                    "repeat": [2, 2, 1]
                }
            }
        arch = architecture
        if not "dilation" in arch["enc"]:
            arch["enc"]["dilation"] = [1]*len(arch["enc"]["repeat"])
        assert len({"first", "enc", "dec"} - {*list(arch.keys())}) == 0, "Missing keys: Need enc, dec, first"
        assert len({"repeat", "width"} - {*list(arch["enc"].keys())}) == 0, "Missing keys enc: Need width, repeat"
        assert len({"repeat", "width"} - {*list(arch["dec"].keys())}) == 0, "Missing keys dec: Need width, repeat"
        assert len(arch["enc"]["repeat"]) == len(arch["enc"]["width"]), "Mismatched dimensions"
        assert len(arch["enc"]["repeat"]) == len(arch["enc"]["dilation"]), "Mismatched dimensions"
        assert len(arch["dec"]["repeat"]) == len(arch["dec"]["width"]), "Mismatched dimensions"
        self.arch = arch
        arch["width"] = arch["enc"]["width"] + arch["dec"]["width"]
        arch_enc_len = len(arch["enc"]["width"])
        arch_dec_len = len(arch["dec"]["width"])

        # Construct Super params (input/output-format, tops etc.)
        super().__init__(
            model_cfg=model_cfg,
            classes=classes,
            last_act=last_act,
            output_format=output_format,
            input_format=input_format,
            input_size=input_size,
            repeat_outputs=arch_dec_len+1 if multi_scale else None
        )

        self.classes = classes
        self.n_classes = len(classes)+1
        self.conv_transpose = conv_transpose
        self.multi_scale = multi_scale
        self.multi_input = multi_input

        # Generate Basic building block & Bigger block
        CBA = self.CBA

        all_blocks = get_all_blocks()
        if type(block) is not list:
            block = [block, block]

        blocks = {}
        for bl, name in zip(block, ["enc", "dec"]):
            if bl not in all_blocks:
                raise ValueError("Block "+bl+" is not a valid block option")
            blocks[name] = all_blocks[bl]

        # Encoder
        bw = first_bw = arch["first"]

        self.input_process = {}
        for key, in_size in zip(self.input_format, self.input_format_sizes):
            self.input_process[key] = CBA(in_size, bw, 3, bn=bn, act=True)
        self.input_process = nn.ModuleDict(self.input_process)

        def get_encoder(wfuse=False):
            prev_bw = arch["first"]
            skips_bw = []
            encoder = []
            fusions = []
            for i, (repeat_block, dilation)  in enumerate(zip(self.arch["enc"]["repeat"], self.arch["enc"]["dilation"])):
                is_last = (i+1 == arch_enc_len)
                if wfuse:
                    fusions.append(FusionModule(
                        model_cfg, in_width=prev_bw, Block=CBA,
                        n_inputs=len(self.input_format) + (0 if i==0 else 1), # Own new branch
                        multi_inputs=None if not multi_input or i==0 else [first_bw]*len(self.input_format),
                    ))
                new_bw = arch["width"][i]
                for j in range(repeat_block):
                    pool = "max" if j+1==repeat_block and not is_last else None
                    drop = small_drop if (not is_last) or j+1<repeat_block else big_drop

                    encoder.append(ConvBlock(
                        model_cfg, blocks["enc"],
                        prev_bw, new_bw,
                        3, bn=bn,
                        pool=pool,
                        conv_transpose=self.conv_transpose,
                        drop=(drop, sddrop),
                        se_ratio=se_ratio,
                        dilation=dilation,
                        first=(i==0) ) )
                    prev_bw = new_bw
                skips_bw.append(prev_bw)
            if wfuse:
                fusions.append(nn.Sequential(
                    FusionModule(
                        model_cfg, in_width=prev_bw,
                        n_inputs=len(self.input_format)+1,
                    ), UpSampling(
                        model_cfg, in_width=prev_bw, width=arch["width"][i+1]
                    )
                ))
                return encoder, skips_bw, fusions
            else:
                return encoder

        self.encoders = []
        for _ in self.input_format:  # all the basic encoders
            self.encoders.append(nn.ModuleList(get_encoder()))
        f_enc, skips_bw, fusions = get_encoder(wfuse=True)  # the fusion encoder
        self.fusions = nn.ModuleList(fusions)
        self.encoders.append(nn.ModuleList(f_enc))
        self.encoders = nn.ModuleList(self.encoders)

        # Decoders (Classif, Pif, Paf...)
        skips_bw.reverse()  # Reverse for easier indexing
        def get_decoder(prev_bw):
            decoder = []
            tops_prev_bw = []
            tops_upsample = []
            for i, repeat_block in enumerate(self.arch["dec"]["repeat"]):
                if self.multi_scale:
                    tops_prev_bw.append(prev_bw)
                    tops_upsample.append(2**(arch_dec_len-i-1))
                is_last = (i+1 == arch_dec_len)
                new_bw = arch["width"][arch_enc_len+i]

                for j in range(repeat_block):
                    pool="up" if not is_last and j+1==repeat_block else None
                    has_skip = j==0
                    concat_width = skips_bw[i+1] if has_skip else None

                    elems = [ConvBlock(
                        model_cfg, blocks["dec"],
                        prev_bw, new_bw,
                        3, bn=bn,
                        concatenate=has_skip,
                        concat_width=concat_width,
                        conv_transpose=self.conv_transpose,
                        drop=(small_drop, sddrop),
                        se_ratio=se_ratio,
                        last_only=True
                    )]
                    prev_bw = new_bw
                    if pool is not None:
                        new_bw = arch["width"][arch_enc_len+i+1] # search for next one
                        elems.append(UpSampling(
                            model_cfg, in_width=prev_bw, width=new_bw
                        ))
                        prev_bw = new_bw
                    decoder.append(nn.Sequential(*elems))
            tops_prev_bw.append(prev_bw)
            tops_upsample.append(1)
            return decoder, tops_prev_bw, tops_upsample

        enc_prev_bw = arch["width"][arch_enc_len]
        decoder_types = {
            "mask": lambda: get_decoder(enc_prev_bw),
            "keypoints": lambda: get_decoder(enc_prev_bw),
            "class": lambda: (
                [SubIdentity()]*(np.sum(arch["dec"]["repeat"])),  # FIXME: not really efficient
                enc_prev_bw
            )
        }
        self.decoders = []
        decoder_tops_prev_bw = []
        decoder_tops_upsample = []
        for out_class in self.inference_output_format:
            decoder, bw_dec, up_dec = decoder_types[out_class.name]()
            self.decoders.append(nn.ModuleList(decoder))
            decoder_tops_prev_bw += bw_dec
            decoder_tops_upsample += up_dec
        self.decoders = nn.ModuleList(self.decoders)

        # Tops
        self.tops = self.make_tops(decoder_tops_prev_bw, decoder_tops_upsample)


    def forward(self, nx):
        all_inputs = []
        # Divide different inputs
        for dtype in self.input_format:
            all_inputs.append(self.input_process[dtype](nx[dtype]))
        nxs = all_inputs

        # Encoder
        encs = [None]*(len(self.arch["enc"]["repeat"])) # save skip connections
        super_ind = 0
        for i, repeat_block in enumerate(self.arch["enc"]["repeat"]):
            encoder = self.encoders[-1]
            if i>0 and self.multi_input:
                all_inputs = [F.avg_pool2d(_in, (2,2)) for _in in all_inputs]
                nxs += all_inputs
            nx = self.fusions[i](nxs)

            ind = super_ind
            for _ in range(repeat_block):
                encs[i], nx = encoder[ind](nx)
                ind += 1
            fuse_nx = nx

            prev_block_last = []
            for encoder, nx in zip(self.encoders[:-1], nxs):
                ind = super_ind
                for _ in range(repeat_block):
                    _, nx = encoder[ind](nx)
                    ind += 1
                prev_block_last.append(nx)
            super_ind = ind
            prev_block_last.append(fuse_nx)
            nxs = prev_block_last


        nx = self.fusions[-1](nxs)
        enc_nx = nx
        encs.reverse()  # reverse for easier indexing

        # Decoders
        decs_nx = []
        for decoder in self.decoders:
            nx = enc_nx
            ind = 0
            for i, repeat_block in enumerate(self.arch["dec"]["repeat"]):
                if self.multi_scale:
                    decs_nx.append(nx)
                for j in range(repeat_block):
                    nx = (nx, encs[i+1]) if j==0 else nx
                    nx = decoder[ind](nx)
                    ind += 1
            decs_nx.append(nx)

        # Tops
        return self.tops(decs_nx)


def fusion_net(**kwargs):
    return FusionNet(**kwargs)
