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
    '[MapReduce(2783069643527250493), Merge(58807196915768375), Stride(5553301806709662180), Merge(8485657051048919630), Share(5308871986), Unfold(3800950797965320760), Unfold(5598271386318413502), Unfold(18155390691884397341), Unfold(380015311295384854), Finalize(1113502945112703776)]',
    '[MapReduce(2783069681112496276), Merge(58801158340947576), Unfold(5598009980104058264), Unfold(5598010025661451717), Finalize(9265162255395717974)]',
    '[MapReduce(2783069643527320833), Merge(58807196915768375), Unfold(5001537661620522554), Unfold(2042468285605536873), Unfold(9136350948553569522), Finalize(15775222193326037076)]',
    '[MapReduce(2783069643526970042), Merge(58801158341257828), Unfold(5598029141215167919), Unfold(5598010025554667668), Finalize(9265184131179475107)]',
    '[MapReduce(2783069681110692612), Unfold(2042468285605532765), Merge(58807196915978551), Split(3800928761808865908), Merge(2632381111369286070), Unfold(3499828444036363936), Unfold(8119037210626728559), Split(5001538560415689975), Merge(9060085369900732680), Finalize(9050284752434443848)]',
    '[MapReduce(2783069681093352481), Merge(58801158341257389), Unfold(5598009980067175591), Unfold(5598009980059254327), Finalize(9265162267339829289)]',
    '[MapReduce(2783069643527051110), Merge(58801158340947576), Unfold(5598010025663018380), Share(5308871985), Merge(8530042273464214974), Unfold(3800753488266645809), Unfold(18198324521478983238), Unfold(7760590068110014270), Finalize(11588118882735684832)]',
    '[MapReduce(2783069681048023296), Merge(58807196915978549), Stride(5553301806689790378), Unfold(5598375390989828167), Unfold(5553302155067586623), Unfold(4539086997643169620), Merge(8530802395730417941), Stride(13903934009476612173), Unfold(4031275876223176495), Finalize(1388604477031682980)]',
    '[MapReduce(2783069643525657196), Merge(58807196916249673), Unfold(5598375391005431542), Unfold(5598364139102120021), Finalize(9279414651678852697)]',
    '[MapReduce(2783069675155637977), Merge(58801158341543776), Unfold(3800731596000359270), Stride(5553364724591851780), Stride(2632387592506072017), Merge(6726352976633773614), Unfold(3517596136683133500), Unfold(5882501606823199572), Unfold(13643519639767482874), Finalize(10070985854545415431)]',
    '[MapReduce(2783069643525497273), Merge(58807196915767455), Unfold(5598375390969426534), Unfold(5598364139062971799), Finalize(9279414645616184566)]',
    '[MapReduce(2783069675155639696), Unfold(2042468285717980327), Share(5308871985), Unfold(104108871466300078), Merge(2632381111599444373), Stride(3490357274531355101), Unfold(1125968334435162869), Unfold(3490357622779562098), Finalize(17724482496467270993)]',
    '[MapReduce(2783069675155749204), Merge(58807196968983587), Unfold(5598375412997386030), Unfold(5598364131412310824), Unfold(5598364479122320061), Finalize(14203694946634020995)]',
    '[MapReduce(2783069643525659678), Unfold(2042468534195016758), Merge(2632387151912106890), Unfold(1126336128374044614), Unfold(3500243490521024394), Finalize(9799724284452345979)]'
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
        kernelPacks, total_flops = sampler.realize(
            model, node, "")
        print(f"kernelPack generated. ")


if __name__ == '__main__':
    test_realize()
