import torch
from torch import nn, Tensor

# KAS
from KAS import Sampler, TreePath, Placeholder, Assembler, Assembled
from KAS.Bindings import CodeGenOptions


class KASGrayConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Placeholder({'N': 5000, 'C_out': 32, 'H': 28, 'W': 28}),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)

        x = image.squeeze(1)
        for layer in self.layers:
            x = layer(x)

        return self.linear(x.view(B, -1))


def conv2d(assembler: Assembler) -> Assembled:
    N, H, W, k, C_out = assembler.get_sizes(
        "N", "H", "W", "k_2", "C_out")

    # Inputs: [N, H, W], [C_out, k_1, k_2]
    in_N, in_H, in_W, out_C, w_k_1, w_k_2 = assembler.make_dims_of_sizes(
        N, H, W, C_out, k, k)
    # [in_N, in_H, in_W, out_C, w_k_1, w_k_2]

    main_H, windows_H = assembler.create_unfold(in_H, k)
    main_W, windows_W = assembler.create_unfold(in_W, k)
    # [in_N, main_H, windows_H, main_W, windows_W, out_C, w_k_1, w_k_2]

    shared_k_1 = assembler.create_share(windows_H, w_k_1)
    shared_k_2 = assembler.create_share(windows_W, w_k_2)
    # [in_N, main_H, main_W, out_C, shared_k_1, shared_k_2]

    in_N.output(0)
    out_C.output(1)
    main_H.output(2)
    main_W.output(3)
    shared_k_1.sum(0)
    shared_k_2.sum(1)

    return assembler.assemble([in_N, in_H, in_W], [out_C, w_k_1, w_k_2])


def test_realize():
    model = KASGrayConv().cuda()
    sampler = Sampler(
        input_shape="[N,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=4096: 1", "H=256", "W=256", "C_out=100"],
        coefficient_specs=["s_1=2", "k_1=3", "k_2=5: 4"],
        seed=0xdeadbeaf,
        depth=8,
        dim_lower=2,
        dim_upper=8,
        save_path='./saves',
        cuda=torch.cuda.is_available(),
        net=model,
        fixed_io_pairs=[(0, 0)],
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )
    assembler = sampler.create_assembler()
    path = conv2d(assembler).convert_to_path(sampler)
    print(path)
    node = sampler.visit(path)


if __name__ == '__main__':
    test_realize()
