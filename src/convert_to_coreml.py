import numpy as np
import math
import torch
from models import generator

import coremltools as ct
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

@register_torch_op
def atan2(context, node):
    inputs = _get_inputs(context, node)
    y = inputs[0]
    x = inputs[1]

    # Add a small value to all zeros in order to avoid division by zero
    epsilon = 1.0e-12
    x = mb.select(cond=mb.equal(x=x, y=0.0), a=mb.add(x=x, y=epsilon), b=x)
    y = mb.select(cond=mb.equal(x=y, y=0.0), a=mb.add(x=y, y=epsilon), b=y)

    angle = mb.select(cond=mb.greater(x=x, y=0.0),
                      a=mb.atan(x=mb.real_div(x=y, y=x)),
                      b=mb.fill(shape=x.shape, value=0.0))

    angle = mb.select(cond=mb.logical_and(x=mb.less(x=x, y=0.0), y=mb.greater_equal(x=y, y=0.0)),
                      a=mb.add(x=mb.atan(x=mb.real_div(x=y, y=x)), y=np.pi),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.less(x=x, y=0.0), y=mb.less(x=y, y=0.0)),
                      a=mb.sub(x=mb.atan(x=mb.real_div(x=y, y=x)), y=np.pi),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.greater(x=y, y=0.0)),
                      a=mb.mul(x=mb.mul(x=0.5, y=np.pi), y=mb.fill(shape=x.shape, value=1.0)),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.less(x=y, y=0.0)),
                      a=mb.mul(x=mb.mul(x=-0.5, y=np.pi), y=mb.fill(shape=x.shape, value=1.0)),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.equal(x=y, y=0.0)),
                      a=mb.fill(shape=x.shape, value=0.0),
                      b=angle)

    context.add(angle, torch_name=node.name)

@register_torch_op
def glu(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    # dim = node.attr["dim"]
    dim = inputs[1].val
    chunks = 2

    total = x.shape[dim]
    size = int(math.ceil(float(total) / float(chunks)))
    split_sizes = [size] * int(math.floor(total / size))
    remainder = total - sum(split_sizes)
    if remainder > 0:
        split_sizes.append(remainder)

    res = mb.split(x=x, split_sizes=split_sizes, axis=dim) #, name=node.name)
    x = res[0]
    gate = res[1]

    x = mb.mul(x=x, y=mb.sigmoid(x=gate))
    context.add(x, torch_name=node.name)

# n_fft = 400
n_fft = 384
# input = torch.rand(1, 2, 321, 201) # 2 seconds
input = torch.rand(1, 2, 334, 193) # 2 seconds
model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).eval()

# checkpoint_path = "best_ckpt/CMGAN-SavedModels-CMGAN_epoch_0_0.084"
checkpoint_path = "/Volumes/ExtremeSSD/MachineLearning/SpeechEnhancement/Experiments/2022-08-02__CMGAN_coreml_189285b__SEDataset0_1/Checkpoints/Mini/CMGAN-SavedModels-2022-08-02__CMGAN_coreml_189285b__SEDataset0_1-CMGAN_epoch_010_0.205"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])

with torch.no_grad():
    traced_model = torch.jit.trace(model, input)
    torch.jit.save(traced_model, "CMGAN.pt")

    ml_model = ct.convert(
      traced_model,
      convert_to="mlprogram",
      inputs=[ct.TensorType(shape=input.shape)])
    ml_model.save("CMGAN.mlpackage")
