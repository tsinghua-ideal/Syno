import torch


def mapping_func_conv(in_size, out_size):
    N, C1, H, W = in_size
    N2, C2, H2, W2 = out_size
    assert N2 == N, f'Batch size change detected: {N} -> {N2}'
    assert H2 == H and W2 == W, f'Not using same padding: {in_size} -> {out_size}'
    mapping = {'N': N, 'C_in': C1, 'C_out': C2, 'H': H, 'W': W}
    return mapping


def mapping_func_gray_conv(in_size, out_size):
    N, H, W = in_size
    N2, C2, H2, W2 = out_size
    assert N2 == N, f'Batch size change detected: {N} -> {N2}'
    assert H2 == H and W2 == W, f'Not using same padding. {in_size} -> {out_size}'
    mapping = {'N': N, 'C_out': C2, 'H': H, 'W': W}
    return mapping


def mapping_func_linear(in_size, out_size):
    N, C1 = in_size
    N2, C2 = out_size
    assert N2 == N, f'Batch size change detected: {N} -> {N2}'
    mapping = {'N': N, 'C_in': C1, 'C_out': C2}
    return mapping
