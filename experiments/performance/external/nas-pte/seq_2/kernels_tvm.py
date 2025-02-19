import tvm
from tvm import relax, te
from tvm.relax import BlockBuilder
from typing import List
import numpy as np

unroll_factor = 16
unrollconv_groups = 2

def weights(C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> List[relax.Constant]:
	weight1 = relax.const(np.random.normal(size=(unroll_factor, C_in, k_1, k_1)).astype("float32"))
	weight2 = relax.const(np.random.normal(size=((C_out - unroll_factor) // unrollconv_groups, (C_in - unroll_factor) // unrollconv_groups, k_1, k_1)).astype("float32"))
	return [weight1, weight2]

def build(bb: BlockBuilder, in_0: relax.Expr, *weights, C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> relax.Var:
	assert C_in > 0 and C_out > 0 and H > 0 and N > 0 and g > 0 and k_1 > 0 and k_2 > 0 and s > 0
	padding = k_1 // 2
	results = bb.emit(
		relax.op.split(
			in_0, [unroll_factor], axis=1
        )
    )
	right = results[1]
	left = bb.emit(
		relax.op.nn.conv2d(
            in_0, weights[0], strides=(1, 1), padding=(padding, padding, padding, padding), dilation=(1, 1), groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32"
        )
    )
	right = bb.emit(
		relax.op.nn.conv2d(
            right, weights[1], strides=(1, 1), padding=(padding, padding, padding, padding), dilation=(1, 1), groups=unrollconv_groups, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32"
        )
    )
	result = bb.emit(
		relax.op.concat([left, right], axis=1)
    )
	return result
