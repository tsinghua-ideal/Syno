import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from tvm import relax
import numpy as np

from Model import KernelMetadata, import_templated_model, substitute_kernels

relax_mod, input_shape = import_templated_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "model_relax"), "ConvNet")

def test_substitute_none():
    result = substitute_kernels(relax_mod, None)
    result.show()

def test_substitute_simple():
    def kernel_builder(kernel_meta: KernelMetadata, bb: relax.BlockBuilder, data: relax.Var) -> relax.Var:
        # This is a simple kernel that adds a constant to the input,
        added = bb.emit(relax.op.add(data, relax.const(kernel_meta.id, "float32")))
        # and performs a 1x1 convolution.
        in_channels = kernel_meta.sinfo_input.shape[1].value
        out_channels = kernel_meta.sinfo_output.shape[1].value
        conv = bb.emit(relax.op.nn.conv2d(
            added,
            relax.const(np.zeros((out_channels, in_channels, 1, 1), "float32")),
            out_dtype="float32",
        ))
        return conv
    result = substitute_kernels(relax_mod, kernel_builder)
    result.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_substitute_none()
    test_substitute_simple()
