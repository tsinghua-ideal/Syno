import tvm
import numpy as np
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

INPUT_SHAPE = (1, 512, 7, 7)

ALL_MAPPINGS = [{'N': 1, 'C_in': 512, 'C_out': 512, 'H': 7}]
metadata = [
    tvm.relax.const(np.random.normal(size=(512, 512, 3, 3)).astype("float32")),
]

@I.ir_module
class Module:
    @R.function
    def main(inp_0: R.Tensor((1, 512, 7, 7), dtype="float32")) -> R.Tensor((1, 512, 7, 7), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((1, 512, 7, 7), dtype="float32") = R.subtract(inp_0, R.const(0, "float32"))
            lv1: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv0, metadata[0], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv2: R.Tensor((1, 512, 7, 7), dtype="float32") = R.exp(lv1)
            gv: R.Tensor((1, 512, 7, 7), dtype="float32") = lv2
            R.output(gv)
        return gv
