import tvm
from tvm import relax, te
from tvm.relax import BlockBuilder
from typing import List
import numpy as np

split_factor = 2

def weights(C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> List[relax.Constant]:
	weights: List[relax.Constant] = [
		relax.const(np.random.normal(size=(C_out, C_in, k_1, k_1)).astype("float32"))
		for _ in range(split_factor)
    ]
	return weights

def build(bb: BlockBuilder, in_0: relax.Expr, *weights, C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> relax.Var:
	assert C_in > 0 and C_out > 0 and H > 0 and N > 0 and g > 0 and k_1 > 0 and k_2 > 0 and s > 0
	padding = k_1 // 2
	break_points = [H // split_factor * (i + 1) for i in range(split_factor - 1)]
	assert len(break_points) > 0
	xs = bb.emit(
		relax.op.split(
			in_0, break_points, axis=2
        )
    )
	results = [
		bb.emit(
			relax.op.nn.conv2d(
				xs[i], weight, strides=(1, 1), padding=(padding, padding, padding, padding), dilation=(1, 1), groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32"
            )
        )
		for i, weight in enumerate(weights)
    ]
	result = bb.emit(
		relax.op.concat(results, axis=2)
    )
	return result
