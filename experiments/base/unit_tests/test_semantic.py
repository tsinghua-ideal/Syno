import torch
from torch import nn

import os, sys, shutil
from types import SimpleNamespace

if os.getcwd() not in sys.path: 
    sys.path.append(os.getcwd())
from models import get_model, KASModel
from models.placeholder import ConvPlaceholder

from KAS import Assembler, Assembled

def get_model_args(model_name: str, batch_size=2, kas_replace_placeholder=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name, 
        batch_size=batch_size, 
        kas_replace_placeholder=kas_replace_placeholder,
        seed=None,
        kas_depth=6,
        kas_min_dim=1,
        kas_max_dim=8,
        kas_max_tensors=3,
        kas_max_reductions=4,
        kas_max_flops=1e9,
        kas_scheduler_cache_dir='.scheduler-cache',
    )

class Impl:
    def __init__(self, assembler: Assembler) -> None:
        self.assembler = assembler
        
    def Conv2d_simple(self) -> Assembled:
        N, H, W, k, C_in, C_out = self.assembler.get_sizes('N', 'H', 'W', 'k', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum(0)
        shared_k_2.sum(1)
        shared_C_in.sum(2)

        return self.assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])
    
    def ConvTranspose2d_simple(self) -> Assembled:
        N, H, W, k, C_in, C_out = self.assembler.get_sizes('N', 'H', 'W', 'k', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum(0)
        shared_k_2.sum(1)
        shared_C_in.sum(2)

        return self.assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])
    
    def Conv2d_dilation(self) -> Assembled:
        N, H, W, k, d, C_in, C_out = self.assembler.get_sizes('N', 'H', 'W', 'k', 'd', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k*d)
        main_W, windows_W = self.assembler.create_unfold(in_W, k*d)
        
        windows_H_strided = self.assembler.create_stride(windows_H, d)
        windows_W_strided = self.assembler.create_stride(windows_W, d)

        shared_k_1 = self.assembler.create_share(windows_H_strided, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W_strided, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum(0)
        shared_k_2.sum(1)
        shared_C_in.sum(2)

        return self.assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])
    
    def Conv2d_group(self) -> Assembled:
        N, H, W, k, g, C_in, C_out = self.assembler.get_sizes('N', 'H', 'W', 'k', 'g', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in / g, k, k)
            
        in_G, in_C_group = self.assembler.create_split(in_C, g)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)
        
        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        final_C_in = self.assembler.create_merge(in_G, shared_C_in)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum(0)
        shared_k_2.sum(1)
        final_C_in.sum(2)

        return self.assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])

    def Conv2d(self) -> Assembled:
        
        N, H, W, k, d, g, C_in, C_out = self.assembler.get_sizes('N', 'H', 'W', 'k', 'd', 'g', 'C_in', 'C_out')
        in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = \
            self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k*s)
        main_W, windows_W = self.assembler.create_unfold(in_W, k*s)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum(0)
        shared_k_2.sum(1)
        shared_C_in.sum(2)

        return self.assembler.assemble('conv', 'in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])

def test_semantic_conv2d(batch_size=4):
    
    model, sampler = get_model(get_model_args('ConvNet', batch_size), return_sampler=True)
    impl = Impl(sampler.create_assembler())
    model.load_kernel(sampler, impl.Conv2d_simple(), 'Conv2d')