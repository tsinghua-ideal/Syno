import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant
from torch.ao.quantization.fake_quantize import FakeQuantize, default_fused_wt_fake_quant, default_fused_act_fake_quant, enable_fake_quant, disable_fake_quant, enable_observer, disable_observer
from KAS import Placeholder
from copy import deepcopy
from functools import partial
import logging

class QuantizedPlaceholder(nn.Module):
    def __init__(self, placeholder: Placeholder):
        super(QuantizedPlaceholder, self).__init__()
        self.placeholder = placeholder
        self.num_weights = len(placeholder.kernel.weights)
        self.quant_weight = nn.ModuleList([FakeQuantize(default_fused_wt_fake_quant) for _ in range(self.num_weights)])
        self.quant_input = FakeQuantize(default_fused_act_fake_quant)
        self.quant_output = FakeQuantize(default_fused_act_fake_quant)

    def forward(self, x):
        x = self.quant_input(x)
        pl = deepcopy(self.placeholder)
        pl.kernel.weights = nn.ParameterList([self.quant_weight[i](self.placeholder.kernel.weights[i]) for i in range(self.num_weights)])
        out = pl(x)
        out = self.quant_output(out)
        return out
    
class QuantizedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super(QuantizedModule, self).__init__()
        self.module = module
        self.quant_weight = FakeQuantize(default_fused_wt_fake_quant)
        self.quant_bias = FakeQuantize(default_fused_wt_fake_quant)
        self.quant_input = FakeQuantize(default_fused_act_fake_quant)
        self.quant_output = FakeQuantize(default_fused_act_fake_quant)

        self.module.weight = nn.Parameter(self.quant_weight(self.module.weight.cpu()).cuda())
        if self.module.bias is not None:
            self.module.bias = nn.Parameter(self.quant_bias(self.module.bias.cpu()).cuda())

    def forward(self, x):
        x = self.quant_input(x)
        out = self.module(x)
        out = self.quant_output(out)
        return out


def replace_placeholders_to_quantized(module: nn.Module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, Placeholder):
            setattr(module, name, QuantizedPlaceholder(child))
            count += 1
        elif isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear) or isinstance(child, nn.BatchNorm2d):
            setattr(module, name, QuantizedModule(child))
            count += 1
        if len(list(child.named_children())) > 0:
            count += replace_placeholders_to_quantized(child)
    return count

def quantize(model: nn.Module, calibration_dataloader: torch.utils.data.DataLoader, num_batches: int):
    replace_placeholders_to_quantized(model)
    model.cuda()
    model.eval()
    model.apply(disable_fake_quant)
    model.apply(enable_observer)
    logging.info("Calibrating......")
    for i, (data, _) in enumerate(calibration_dataloader):
        model(data.cuda())
    logging.info("Calibration Finished.")
    model.apply(disable_observer)
    model.apply(enable_fake_quant)
