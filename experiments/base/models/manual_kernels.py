from KAS import Sampler, Assembled

class ManualImpl:
    def __init__(self, sampler: Sampler) -> None:
        self.assembler = sampler.create_assembler()

    def Conv2d_simple(self) -> Assembled:
        N, H, W, k, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1, w_k_2],
        )

    def Conv2d_dilation(self) -> Assembled:
        N, H, W, k_1, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "s", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k_1, k_1)

        main_H, windows_H = self.assembler.create_unfold(in_H, k_1 * k_1)
        main_W, windows_W = self.assembler.create_unfold(in_W, k_1 * k_1)

        windows_H_strided = self.assembler.create_stride(windows_H, k_1)
        windows_W_strided = self.assembler.create_stride(windows_W, k_1)

        shared_k_1 = self.assembler.create_share(windows_H_strided, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W_strided, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1, w_k_2],
        )

    def Conv2d_group(self) -> Assembled:
        N, H, W, k_1, g, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "s", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C_group,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out / g, C_in / g, k, k)
        
        # Spatial dimensions
        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)

        # channel dimensions
        in_G, in_C_group = self.assembler.create_split(in_C, C_in / g)

        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        
        tmp_dim = self.assembler.create_expand(C_out / g)
        out_C_group_masked = self.assembler.create_share(tmp_dim, out_C_group)
        final_C_out = self.assembler.create_merge(in_G, out_C_group_masked)

        in_N.output(0)
        final_C_out.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W, tmp_dim],
            [out_C_group, w_in_C, w_k_1, w_k_2],
        )
    
    def Conv2d_group_oas(self) -> Assembled:
        N, H, W, k_1, g, r, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "g", "s", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_G, 
            out_C_group,
            w_in_C,
            w_k_1,
            w_k_2,
            w_interm_out_G,
            w_interm_out_C_group, 
            out_C, 
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, g, C_out / g / r, C_in / g, k, k, g, C_out / g / r, C_out)
        
        # Spatial dimensions
        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)

        # channel dimensions
        in_G, in_C_group = self.assembler.create_split(in_C, C_in / g)

        shared_G = self.assembler.create_share(in_G, out_G)
        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        
        tmp_dim = self.assembler.create_expand(C_out / g / r)
        out_C_group_masked = self.assembler.create_share(tmp_dim, out_C_group)
        interm_G_contracted = self.assembler.create_share(shared_G, w_interm_out_G)
        interm_C_out_group_contracted = self.assembler.create_share(out_C_group_masked, w_interm_out_C_group)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()
        interm_G_contracted.sum()
        interm_C_out_group_contracted.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1 * in_2",
            [in_N, in_C, in_H, in_W, tmp_dim],
            [out_G, out_C_group, w_in_C, w_k_1, w_k_2],
            [out_C, w_interm_out_G, w_interm_out_C_group]
        )
    
    def Conv2d_group_oas_same(self) -> Assembled:
        N, H, W, k_1, g, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "g", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_G, 
            out_C_group,
            w_in_C,
            w_k_1,
            w_k_2,
            w_interm_out_C, 
            out_C, 
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, g, C_out / g, C_in / g, k, k, C_out, C_out)
        
        # Spatial dimensions
        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)

        # channel dimensions
        in_G, in_C_group = self.assembler.create_split(in_C, C_in / g)

        shared_G = self.assembler.create_share(in_G, out_G)
        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        
        tmp_dim = self.assembler.create_expand(C_out / g)
        out_C_group_masked = self.assembler.create_share(tmp_dim, out_C_group)
        interm_C_out = self.assembler.create_merge(shared_G, out_C_group_masked)
        
        interm_C_out_contracted = self.assembler.create_share(interm_C_out, w_interm_out_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()
        interm_C_out_contracted.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1 * in_2",
            [in_N, in_C, in_H, in_W, tmp_dim],
            [out_G, out_C_group, w_in_C, w_k_1, w_k_2],
            [out_C, w_interm_out_C]
        )

    def Conv2d_group_oas_double(self) -> Assembled:
        N, H, W, k_1, g, r, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "g", "s", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_G, 
            out_C_group,
            w_in_C,
            w_k_1,
            w_k_2,
            w_interm_out_C, 
            out_C, 
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, g, C_out / g * r, C_in / g, k, k, C_out * r, C_out)
        
        # Spatial dimensions
        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)

        # channel dimensions
        in_G, in_C_group = self.assembler.create_split(in_C, C_in / g)

        shared_G = self.assembler.create_share(in_G, out_G)
        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        
        tmp_dim = self.assembler.create_expand(C_out / g * r)
        out_C_group_masked = self.assembler.create_share(tmp_dim, out_C_group)
        interm_C_out = self.assembler.create_merge(shared_G, out_C_group_masked)
        
        interm_C_out_contracted = self.assembler.create_share(interm_C_out, w_interm_out_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()
        interm_C_out_contracted.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1 * in_2",
            [in_N, in_C, in_H, in_W, tmp_dim],
            [out_G, out_C_group, w_in_C, w_k_1, w_k_2],
            [out_C, w_interm_out_C]
        )
        
    def Conv2d_FC(self) -> Assembled:
        N, H, W, k, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "s", "C_in", "C_out"
        )
        ratio = s * s
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
            interm_out_C, 
            w_interm_fc, 
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k, C_out / ratio, C_out / ratio)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)
        
        shared_interm_C_in = self.assembler.create_share(interm_out_C, w_interm_fc)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()
        shared_interm_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1 * in_2",
            [in_N, in_C, in_H, in_W],
            [interm_out_C, w_in_C, w_k_1, w_k_2],
            [out_C, w_interm_fc],
        )
        
    def Conv2d_pool(self) -> Assembled:
        N, H, W, k_1, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "s", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)
        
        # pool along spatial dimensions
        H_pooled, s_H = self.assembler.create_split(in_H, s)
        W_pooled, s_W = self.assembler.create_split(in_W, s)
        
        # Convolutions
        main_H_pooled, windows_H = self.assembler.create_unfold(H_pooled, k)
        main_W_pooled, windows_W = self.assembler.create_unfold(W_pooled, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        # unpool along spatial dimensions
        
        s_H_expand = self.assembler.create_expand(s)
        s_W_expand = self.assembler.create_expand(s)
        main_H = self.assembler.create_merge(main_H_pooled, s_H_expand)
        main_W = self.assembler.create_merge(main_W_pooled, s_W_expand)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        s_H.sum()
        s_W.sum()
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W, s_H_expand, s_W_expand],
            [out_C, w_in_C, w_k_1, w_k_2],
        )
        
    def Conv2d_pool1d(self) -> Assembled:
        N, H, W, k_1, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "s", "C_in", "C_out"
        )
        k = k_1
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)
        
        # pool along spatial dimensions
        H_pooled, s_H = self.assembler.create_split(in_H, s)
        main_H_pooled, windows_H = self.assembler.create_unfold(H_pooled, k)
        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)
        
        main_W, windows_W = self.assembler.create_unfold(in_W, k)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)

        s_H_expand = self.assembler.create_expand(s)
        main_H = self.assembler.create_merge(main_H_pooled, s_H_expand)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        s_H.sum()
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W, s_H_expand],
            [out_C, w_in_C, w_k_1, w_k_2],
        )
        
    def Conv1d_shift1d(self) -> Assembled:
        N, H, W, k, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W = self.assembler.create_shift(in_W, 1)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1],
        )

    def Shift2d(self) -> Assembled:
        N, H, W, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in)
        
        # Convolutions
        main_H = self.assembler.create_shift(in_H, 1)
        main_W = self.assembler.create_shift(in_W, 1)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C],
        )
