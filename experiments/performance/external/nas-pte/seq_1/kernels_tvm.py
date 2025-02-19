import tvm
from tvm import relax, te
from tvm.relax import BlockBuilder
from typing import List
import numpy as np

groups = [2, 4]

def weights(C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> List[relax.Constant]:
	split_factor = len(groups)
	weights: List[relax.Constant] = [
		relax.const(np.random.normal(size=(C_out // split_factor // g, C_in // g, k_1, k_1)).astype("float32"))
		for g in groups
    ]
	return weights

def build(bb: BlockBuilder, in_0: relax.Expr, *weights, C_in: int = -1, C_out: int = -1, H: int = -1, N: int = -1, g: int = 32, k_1: int = 3, k_2: int = 7, s: int = 2, ) -> relax.Var:
	assert C_in > 0 and C_out > 0 and H > 0 and N > 0 and g > 0 and k_1 > 0 and k_2 > 0 and s > 0
	split_factor = len(groups)
	padding = k_1 // 2
	results = [
		bb.emit(
			relax.op.nn.conv2d(
				in_0, weight, strides=(1, 1), padding=(padding, padding, padding, padding), dilation=(1, 1), groups=g, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32"
            )
        )
		for g, weight in zip(groups, weights)
    ]
	result = bb.emit(
		relax.op.concat(results, axis=1)
    )
	return result
