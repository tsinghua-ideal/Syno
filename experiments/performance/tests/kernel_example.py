from tvm import auto_scheduler, te

N, H, W, CO, CI = 16, 28, 28, 96, 64
input_shape = (N, CI, H, W)
output_shape = (N, CO, H, W)

K, S = 3, 2
weights_shapes = [(CO, CI, K)]

kernel_hash = 12345
kernel_placeholder_index = 0
kernel_name = f"kas_kernel_{kernel_hash}_{kernel_placeholder_index}"

@auto_scheduler.register_workload(func_name=kernel_name)
def kernel_workload():
    in_0 = te.placeholder((N, CI, H, W), name="in_0", dtype="float32")
    in_1 = te.placeholder((CO, CI, K), name="in_1", dtype="float32")
    r_0 = te.reduce_axis((0, CI), name="r_0")
    r_1 = te.reduce_axis((0, K), name="r_1")
    out = te.compute(
        (N, CO, H, W),
        lambda n, co, h, w: te.sum(
            te.if_then_else(
                te.all(h + r_1 >= K // 2, h + r_1 < H + K // 2),
                in_0[
                    n,
                    r_0,
                    te.indexmod(h + r_1 - K // 2 + 1, H),
                    S * te.indexmod(te.indexmod(w, W // S) + 1, W // S) + te.indexdiv(h, W // S)
                ],
                0.0,
            ) * in_1[co, r_0, r_1],
            axis=[r_0, r_1],
        ),
        name="out",
    )
    return [in_0, in_1, out]
