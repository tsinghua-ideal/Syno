def dense(assembler):
    N, C_in, C_out = assembler.get_sizes('N', 'C_in', 'C_out')

    # Inputs: [N, C_in], [C_in, C_out]
    in_N, in_C, w_in_C, w_out_C = assembler.make_dims_of_sizes(N, C_in, C_in, C_out)
    # [in_N, in_C, w_in_C, w_out_C]

    shared_C_in = assembler.create_share(in_C, w_in_C)
    # [in_N, shared_C_in, w_out_C]

    in_N.output(0)
    w_out_C.output(1)
    shared_C_in.mean(0)

    return assembler.assemble('in_0 * in_1', [in_N, in_C], [w_in_C, w_out_C])
