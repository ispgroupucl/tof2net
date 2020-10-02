# Largely inspired by:
# https://github.com/Lyken17/pytorch-OpCounter
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd


def check_inputs(inputs):
    """ Checks that inputs isn't composed of multiple tensors """
    iitype = type(inputs)
    if iitype == tuple:
        if len(inputs)>1:
            print(f"Input longer than 1 are not supported")
            raise NotImplementedError()
        inputs = inputs[0]
    elif iitype != torch.Tensor:
        raise NotImplementedError()
    return inputs

def count_conv2d(m:nn.Conv2d, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a conv2d layer """
    inputs = check_inputs(inputs)
    cin  = m.in_channels
    cout = m.out_channels
    x_stride, y_stride = m.stride
    x_kernel, y_kernel = m.kernel_size

    total_ops = cin*cout*x_kernel*y_kernel * (inputs.size(-2)/x_stride) * (inputs.size(-1)/y_stride) / m.groups * 2

    m.total_ops = torch.Tensor([int(total_ops)])

def count_bn(m:nn.BatchNorm2d, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a bn2d layer """
    inputs = check_inputs(inputs)
    cin  = m.num_features
    total_ops = cin * inputs.size(-2) * inputs.size(-1) * 4
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m:nn.Linear, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a linear layer """
    cin  = m.in_features
    cout = m.out_features
    total_ops = cin*cout
    m.total_ops = torch.Tensor([int(total_ops)])


def profile(model, cfg=None, dims=None, logpath=None):
    """ Counts the #params and #ops in a torch model

    Arguments:
        model (torch.nn.Module): the neural network to be evaluated
        cfg (dict): the config used to determine the input size for computation
        dims (tuple): if cfg is None, dims is used to determine niput size
        logpath (Path): where to store a models onnx representation for easier porting
    """
    assert not (cfg is None and dims is None), "either set cfg or dims"
    handler_collection = []
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            raise Warning("Either .total_ops or .total_params is already defined in %s.\n"
                          "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        fn = None
        if isinstance(m, nn.Conv2d):
            fn = count_conv2d
        elif isinstance(m, nn.BatchNorm2d):
            fn = count_bn
        elif isinstance(m, nn.Linear):
            fn = count_linear

        if fn is not None:
            # print("Registered: ", m_type)
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        # else:
        #     print("Skipped: ", m_type)

    model.eval()
    model.apply(add_hooks)
    if dims is None:
        dims = tuple(cfg["dataset"]["dimensions"])
        if not cfg["dataset"].get("channels_first", False):
            dims = (dims[2], dims[0], dims[1])

    x = {}
    for dtype, dtype_bw in zip(model.input_format, model.input_format_sizes):
        input_size = (1,dtype_bw,) + dims[-2:]
        x[dtype] = torch.rand(input_size)

    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params
        del m.total_ops
        del m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    for handler in handler_collection:
        handler.remove()

    n_params = total_params/10**6
    n_ops = int(total_ops)/10**9
    info = "===================\n" +\
            "This model contains:\n #PARAMS = {:.3f} M \n #GOP    = {:.3f}\n".format(n_params, n_ops) +\
            "===================\n"

    # Export the model
    if logpath is not None:
        logpath.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            model.eval()
            torch.onnx.export(model,                   # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            logpath/"skeleton.onnx",   # where to save the model (can be a file or file-like object)
                            export_params=False,       # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=False, # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                            'output' : {0 : 'batch_size'}})


    return total_ops, total_params, info

