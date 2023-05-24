import torch
from torch import nn, Tensor

# KAS
from KAS import Sampler, TreePath, Placeholder
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


Error_paths = [
    '[MapReduce(2783069643527250467), Stride(2632381837011155637), Merge(6726352459680335266), Share(5308871985), Stride(5837519633673251446), Unfold(5837542417638771960), Unfold(5882446372082621335), Finalize(17846091640672260811)]',
    '[MapReduce(2043871597405884268), Stride(2632387592528325923), Merge(6726352976664150620), Unfold(3517596134328398467), Share(5308871985), Stride(5837766385816657548), Merge(8204577194128690212), Unfold(16304323359660244360), Unfold(16304301339652712572), Finalize(3530661378453405124)]',
    '[MapReduce(2782943649840858737), Stride(2632381837011126851), Unfold(2042468534195013123), Merge(6726352458243962039), Share(5308871986), Stride(5837519675137739879), Share(5308871985), Unfold(5837542048951196897), Unfold(3517724437682295378), Finalize(11522634464546934446)]',
    '[MapReduce(2783069643527041936), Merge(58801158341257389), Stride(5553499391242419359), Split(2042468534195021024), Split(4111118744132176007), Stride(5316294912210705916), Unfold(5598219357826875174), Unfold(9573503677142918491), Merge(13633664612953374117), Finalize(9530821335105833530)]',
    '[MapReduce(2783069681088653711), Stride(2632387592506554857), Share(5308871985), Merge(6726352976633499930), Unfold(5882486970370335152), Stride(5837766383390927106), Split(350495520958), Unfold(5837789291091627351), Unfold(307201288018071313), Finalize(1252656416876091858)]',
    '[MapReduce(2783069681110662043), Merge(58807196915978614), Unfold(5598375388446195329), Stride(5553307875349481238), Unfold(2042468285605543600), Merge(8485651060330708368), Stride(301465624095599848), Unfold(4395742347393419619), Unfold(18155531718437544065), Finalize(8344533731903383784)]'
]


def test_realize():
    model = KASGrayConv().cuda()
    sampler = Sampler(
        input_shape="[N,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=4096: 1", "H=256", "W=256", "C_out=100"],
        coefficient_specs=["s_1=2", "k_1=3", "k_2=5"],
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
    for path_repr in Error_paths:
        path = TreePath.decode_str(path_repr)
        print(path)
        node = sampler.visit(path)
        print(node.is_final())
        print(f"Finalize detected. ")


if __name__ == '__main__':
    test_realize()
