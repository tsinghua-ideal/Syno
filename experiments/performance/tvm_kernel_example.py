import tvm
from tvm import relax, te
from tvm.relax import BlockBuilder
import numpy as np


def build(bb: BlockBuilder, in_0: relax.Var, N: int = 1, C_in: int = -1, C_out: int = -1, H: int = -1, W: int = -1, k: int = 3, s: int = 2) -> relax.Var:
    assert N > 0 and C_in > 0 and C_out > 0 and H > 0 and W > 0 and k > 0 and s > 0
    in_1: relax.Constant = relax.const(np.random.normal(size=(C_out, C_in, k, k)).astype("float32"))
    def subgraph_0(in_0: te.Tensor) -> te.Tensor:
        ri_0 = te.reduce_axis((0, s), "ri_0")
        ri_1 = te.reduce_axis((0, s), "ri_1")
        return te.compute(
            (N, C_in, H // s, W // s),
            lambda i_0, i_1, i_2, i_3:
                te.sum(
                    in_0[i_0, i_1, i_2 * s + ri_0, i_3 * s + ri_1],
                    axis=[ri_0, ri_1],
                ),
            name="subgraph_0",
        )
    def subgraph_1(in_0: te.Tensor, in_1: te.Tensor) -> te.Tensor:
        ri_0 = te.reduce_axis((0, k), "ri_0")
        ri_1 = te.reduce_axis((0, k), "ri_1")
        ri_2 = te.reduce_axis((0, C_in), "ri_2")
        return te.compute(
            (N, C_out, H // s, W // s),
            lambda i_0, i_1, i_2, i_3:
                te.sum(
                    te.if_then_else(
                        te.all(i_2 + ri_0 - k // 2 >= 0, i_2 + ri_0 - k // 2 < k, i_3 + ri_1 - k // 2 >= 0, i_3 + ri_1 - k // 2 < k),
                        in_0[i_0, ri_2, i_2 + ri_0 - k // 2, i_3 + ri_1 - k // 2] * in_1[i_1, ri_2, ri_0, ri_1],
                        0.0,
                    ),
                    axis=[ri_0, ri_1, ri_2],
                ),
            name="subgraph_1",
        )
    def subgraph_2(in_0: te.Tensor) -> te.Tensor:
        return te.compute(
            (N, C_out, H, W),
            lambda i_0, i_1, i_2, i_3:
                in_0[i_0, i_1, te.indexdiv(i_2, s), te.indexdiv(i_3, s)],
            name="subgraph_2",
        )
    res_subgraph_0 = bb.emit_te(subgraph_0, in_0)
    res_subgraph_1 = bb.emit_te(subgraph_1, res_subgraph_0, in_1)
    res_subgraph_2 = bb.emit_te(subgraph_2, res_subgraph_1)
    return res_subgraph_2
