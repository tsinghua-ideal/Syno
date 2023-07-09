import torch
import torch.nn.functional as F
from typing import Dict
from torch import nn
from KAS import Placeholder


# TODO: statistics for placeholder
class LinearPlaceholder(Placeholder):
    def __init__(self, in_features, out_features) -> None:
        super(LinearPlaceholder, self).__init__(
            refered_layer=nn.Linear(in_features, out_features, bias=False),
            mapping_func=LinearPlaceholder.mapping
        )

    @staticmethod
    def impl(assembler):
        N, C_in, C_out = assembler.get_sizes('N', 'C_in', 'C_out')
        in_N, in_C, w_in_C, w_out_C = assembler.make_dims_of_sizes(N, C_in, C_in, C_out)
        
        shared_C_in = assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        w_out_C.output(1)
        shared_C_in.mean(0)

        return assembler.assemble('linear', 'in_0 * in_1', [in_N, in_C], [w_in_C, w_out_C])

    @staticmethod
    def mapping(in_size, out_size):
        n, in_features = in_size
        n2, out_features = out_size
        assert n == n2
        return {'N': n, 'C_in': in_features, 'C_out': out_features}


class ConvGen(nn.Module):
    def __init__(self, ic, oc, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(oc, ic, k, k))
        nn.init.trunc_normal_(self.w, std=.1)

    def forward(self, x):
        n, c, h, w = x.size()
        x = F.unfold(x, (self.k, 1), padding=((self.k - 1) // 2, 0))
        x = x.view(n, c * self.k, h, w)
        x = F.unfold(x, (1, self.k), padding=(0, (self.k - 1) // 2))
        x = x.view(n, c, self.k, self.k, h, w)
        return torch.einsum('nxijab,yxij->nyab', x, self.w) / (self.k ** 2)


class ConvPlaceholder(Placeholder):
    def __init__(self, in_features, out_features, kernel_size) -> None:
        super(ConvPlaceholder, self).__init__(
            refered_layer=nn.Conv2d(in_features, out_features, kernel_size, bias=False, padding='same'),
            # refered_layer=ConvGen(in_features, out_features, kernel_size),
            mapping_func=ConvPlaceholder.mapping
        )

    @staticmethod
    def impl(assembler):
        N, H, W, k, C_in, C_out = assembler.get_sizes('N', 'H', 'W', 'k', 'C_in', 'C_out')
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
        shared_k_1.mean(0)
        shared_k_2.mean(1)
        shared_C_in.mean(2)

        return assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])
    
    @staticmethod
    def mapping(in_size, out_size):
        n, c1, h, w = in_size
        n2, c2, h2, w2 = out_size
        assert n == n2
        return {'N': n, 'C_in': c1, 'C_out': c2, 'H': h, 'W': w}
    