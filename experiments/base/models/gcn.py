import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from KAS import init_weights

from .model import KASModel
from .placeholder import GNNLinearPlaceholder


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = GNNLinearPlaceholder(in_channels, out_channels)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.apply(init_weights)
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(KASModel):
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 16)
        self.conv_2 = GCNConv(16, 7)

    def profile(self, batch_size=1, force_update=False, not_count_placeholder=False, seq_len=None):
        return 0, 0
    
    def sample_input_shape(self, seq_len):
        assert False

    def sampler_parameters(self, args=None):
         return {
            'input_shape': '[N, H_in: unordered]',
            'output_shape': '[N, H_out: unordered]',
            'primary_specs': ['N: 0', 'H_in: 2', 'H_out: 2'],
            'coefficient_specs': ['k_1=2: 2', 'k_2=3: 2', 'k_3=5: 2', 'g=32: 2'],
            'fixed_io_pairs': [(0, 0)],
        }

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_2(x, edge_index)

        return F.log_softmax(x, dim=1)
