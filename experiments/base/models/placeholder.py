import torch
import torch.nn.functional as F
from typing import Dict
from torch import nn
from KAS import Placeholder


class LinearPlaceholder(Placeholder):
    def __init__(self, in_features, out_features) -> None:
        super(LinearPlaceholder, self).__init__(
            referred_layer=nn.Linear(in_features, out_features, bias=False),
            mapping_func=LinearPlaceholder.mapping
        )

    @staticmethod
    def impl(assembler):
        N, seq_len, H_in, H_out = assembler.get_sizes('N', 'seq_len', 'H_in', 't*H_in')
        in_N, in_seq_len, in_H_in, w_H_in, w_H_out = assembler.make_dims_of_sizes(N, seq_len, H_in, H_in, H_out)
        
        shared_H_in = assembler.create_share(in_H_in, w_H_in)

        in_N.output(0)
        in_seq_len.output(1)
        w_H_out.output(2)
        shared_H_in.sum()

        return assembler.assemble('linear', 'in_0 * in_1', [in_N, in_seq_len, in_H_in], [w_H_in, w_H_out])

    @staticmethod
    def mapping(in_size, out_size):
        n, seq_len, in_features = in_size
        n2, seq_len2, out_features = out_size
        assert n == n2 and seq_len == seq_len2
        return {'N': n, 'seq_len': seq_len, 'H_in': in_features}


class ConvPlaceholder(Placeholder):
    def __init__(self, in_features, out_features, kernel_size) -> None:
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            kernel_size = kernel_size[0]
        super(ConvPlaceholder, self).__init__(
            referred_layer=nn.Conv2d(in_features, out_features, kernel_size, bias=False, padding=kernel_size // 2),
            mapping_func=ConvPlaceholder.mapping
        )

    @staticmethod
    def impl(assembler):
        N, H, W, k, C_in, C_out = assembler.get_sizes('N', 'H', 'W', 'k_1', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = assembler.create_unfold(in_H, k)
        main_W, windows_W = assembler.create_unfold(in_W, k)

        shared_k_1 = assembler.create_share(windows_H, w_k_1)
        shared_k_2 = assembler.create_share(windows_W, w_k_2)
        shared_C_in = assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])
    
    @staticmethod
    def mapping(in_size, out_size):
        n, c1, h, w = in_size
        n2, c2, h2, w2 = out_size
        assert n == n2
        return {'N': n, 'C_in': c1, 'C_out': c2, 'H': h, 'W': w}
    
    @staticmethod
    def exclusion_condition(in_size, out_size) -> bool:
        n, c1, h, w = in_size
        n2, c2, h2, w2 = out_size
        return not (h >= 4 and w >= 4 and c1 >= 4 and c2 >= 64)