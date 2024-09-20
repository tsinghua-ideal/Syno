import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant
from KAS import Placeholder
from copy import deepcopy

class QuantizedPlaceholder(nn.Module):
    def __init__(self, placeholder: Placeholder):
        super(QuantizedPlaceholder, self).__init__()
        self.placeholder = placeholder
        self.num_weights = len(placeholder.kernel.weights)
        self.quant_weight = nn.ModuleList([quant.FakeQuantize(quant.default_weight_fake_quant) for _ in range(self.num_weights)])
        self.quant_input = quant.FakeQuantize(quant.default_fake_quant)
        self.quant_output = quant.FakeQuantize(quant.default_fake_quant)

    def forward(self, x):
        x = self.quant_input(x)
        pl = deepcopy(self.placeholder)
        pl.kernel.weights = nn.ParameterList([self.quant_weight[i](self.placeholder.kernel.weights[i]) for i in range(self.num_weights)])
        out = pl(x)
        out = self.quant_output(out)
        return out


class QuantizedConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d):
        super(QuantizedConv2d, self).__init__()
        self.module = module
        self.quant_weight = quant.FakeQuantize(quant.default_weight_fake_quant)
        self.quant_bias = quant.FakeQuantize(quant.default_weight_fake_quant)
        self.quant_input = quant.FakeQuantize(quant.default_fake_quant)
        self.quant_output = quant.FakeQuantize(quant.default_fake_quant)

    def forward(self, x):
        x = self.quant_input(x)
        w = self.quant_weight(self.module.weight)
        b = self.quant_bias(self.module.bias) if self.module.bias else None
        out = F.conv2d(x, w, b, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
        out = self.quant_output(out)
        return out

def replace_placeholders_to_quantized(module: nn.Module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, Placeholder):
            setattr(module, name, QuantizedPlaceholder(child))
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantizedConv2d(child))
        if len(list(child.named_children())) > 0:
            count += replace_placeholders_to_quantized(child)
    return count

def quantize(model: nn.Module):
    replace_placeholders_to_quantized(model)
    model.cuda()
