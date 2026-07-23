"""Custom ONNX symbolic overrides so TorchANI's AEV/neighborlist ops export.

During a fixed-size trace, ops like triu_indices have constant integer args, so
we can bake the resulting index tensors as ONNX Constants. Consequence: an
exported graph is specialized to the traced atom count (one ONNX per size).
"""
import torch
from torch.onnx import symbolic_helper


@symbolic_helper.parse_args("i", "i", "i", "i", "i", "v", "b")
def triu_indices(g, row, col, offset, dtype, layout, device, pin_memory):
    idx = torch.triu_indices(row, col, offset).to(torch.int64)
    return g.op("Constant", value_t=idx)


def register():
    torch.onnx.register_custom_op_symbolic("aten::triu_indices", triu_indices, 17)
