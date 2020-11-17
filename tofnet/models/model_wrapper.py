import warnings
from tofnet.models.blocks import get_last_act, create_cba
from tofnet.utils.outputs import parse_outputs
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools


class ModelWrapper(nn.Module):
    """ To avoid code-duplication, this module handles the creation of the
        Conv-Bn-Act module (CBA) as well as the input- and output-names processing.
        It also handles the final layer of the network's outputs/heads
    """
    def __init__(self, model_cfg, classes=None, last_act="linear",
            input_format=None, output_format=None, input_size=None,
            repeat_outputs=None):
        super().__init__()

        # Default input-output formats
        if input_format is None:
            input_format = ["image"]
        if type(input_format) is not list:
            input_format = [input_format]
        inf = [x.split("_") for x in input_format]
        self.input_format_sizes = []
        self.input_format = []
        for x in input_format:
            try:
                sx = x.split("_")
                bw = int(x[-1])
                x_dtype = "_".join(sx[:-1])
            except ValueError:
                bw = 1 if input_size is None else input_size[0]
                x_dtype = x
            finally:
                self.input_format_sizes.append(bw)
                self.input_format.append(x_dtype)


        if output_format is None:
            output_format = ["mask"]
        default_output_size = {
            "class": classes,  # FIXME: Should not be necessary anymore
            "mask": classes
        }
        inference_output_format = output_format
        if repeat_outputs is not None:
            output_format = []
            for of in inference_output_format:
                if "class" not in of:
                    output_format += [of]*repeat_outputs
                else:
                    output_format.append(of)

        self.output_format = parse_outputs(output_format, **default_output_size)
        self.inference_output_format = parse_outputs(inference_output_format, **default_output_size)

        # Generate Basic building block & Bigger block
        self.CBA = create_cba(model_cfg)

        # Get final activation layer
        self.last_act = get_last_act(last_act)
        self.outputs = self.output_format

    def make_tops(self, prev_bw_list, upscale_list=None):
        """

        Arguments:
             prev_bw_list: is a list of the previous widths (aka #FM)
                        MUST have the same size as self.output_format
             upscale_list: is a list of the amount of upscaling needed
                        MUST have the same size as self.output_format

        """
        assert len(prev_bw_list) == len(self.output_format)
        if upscale_list is None:
            upscale_list = [1]*len(self.output_format)
        assert len(upscale_list) == len(self.output_format)

        CBA = self.CBA
        # Tops
        top_types = {
            "mask": lambda prev_bw, new_bw, up_ratio: nn.Sequential(
                nn.ReLU(),
                CBA(prev_bw, new_bw, 1, bn=False, act=False),
                *([nn.UpsamplingBilinear2d(scale_factor=up_ratio)] if up_ratio > 1 else [])
            ), "keypoints": lambda prev_bw, new_bw, up_ratio: nn.Sequential(
                nn.ReLU(),
                CBA(prev_bw, new_bw, 1, bn=False, act=False),
                *([nn.UpsamplingBilinear2d(scale_factor=up_ratio)] if up_ratio > 1 else [])
            ), "class": lambda prev_bw, new_bw, _: nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                CBA(prev_bw, new_bw, 1, bn=False, act=False, bias=False)
            )  # Equivalent to a Linear layer
        }

        self._tops = []
        for i, (out_class, prev_bw, up_ratio) in enumerate(zip(self.output_format, prev_bw_list, upscale_list)):
            self._tops.append(top_types[out_class.name](prev_bw, out_class.n_outputs, up_ratio))
        self._tops = nn.ModuleList(self._tops)
        return self._forward_tops


    def _forward_tops(self, decs_nx_list):
        # Tops
        final_nx = tuple()
        for top, nx, out_class in zip(self._tops, decs_nx_list, self.output_format):
            out_type = out_class.name
            nx = self.last_act(top(nx))
            if out_type in {"class"}:
                nx = nx.squeeze_(-1).squeeze_(-1)
            elif out_type in {"mask"}: # TODO: check if really necessary
                                        # Should we add keypoints too??
                if out_class.n_outputs == 1:
                    nx = nx.squeeze_(1)
            final_nx += (nx,)

        return final_nx