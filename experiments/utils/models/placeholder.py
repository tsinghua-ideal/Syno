from torch import nn
from KAS import Placeholder


class DensePlaceholder(Placeholder):
    def __init__(self, in_features, out_features) -> None:
        super(DensePlaceholder, self).__init__(
            refered_layer=nn.Linear(in_features, out_features, bias=False),
            mapping_func=DensePlaceholder.mapping
        )

    @staticmethod
    def impl(assembler):
        N, C_in, C_out = assembler.get_sizes('N', 'C_in', 'C_out')
        in_N, in_C, w_in_C, w_out_C = assembler.make_dims_of_sizes(N, C_in, C_in, C_out)
        
        shared_C_in = assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        w_out_C.output(1)
        shared_C_in.mean(0)

        return assembler.assemble('in_0 * in_1', [in_N, in_C], [w_in_C, w_out_C])

    @staticmethod
    def mapping(in_size, out_size):
        n, in_features = in_size
        n2, out_features = out_size
        assert n == n2
        return {'N': n, 'C_in': in_features, 'C_out': out_features}
