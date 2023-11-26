import torch
import thop
from collections import OrderedDict
from torch import nn
from typing import List, Tuple, Union, Optional
from os import PathLike
import KAS
from KAS import Placeholder, KernelLoader

from .placeholder import (
    LinearPlaceholder,
    ConvPlaceholder,
    ViTLinearPlaceholder,
    ConvNeXtLinearPlaceholder,
)


class KASModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flops = 0
        self.params = 0

    def load_kernel(
        self, kernel: KernelLoader, compile=False, batch_size=1, seq_len=None
    ) -> str:
        kernel_packs = kernel.construct_kernel_packs()
        placeholders = KAS.Sampler._extract_placeholders(self)
        assert (
            len(placeholders) == kernel.get_count_placeholders()
        ), f"Kernel {kernel} has {kernel.get_count_placeholders()} placeholders, but {len(placeholders)} placeholders are found in the model"
        flops = []
        for i, (placeholder, kernel_pack) in enumerate(zip(placeholders, kernel_packs)):
            placeholder.reload(kernel_pack, False)
            placeholder.referred_layer = None
            placeholder.set_flops(kernel.get_flops(i))
            placeholder.set_params(
                sum(weight.numel() for weight in kernel_pack.weights)
                if hasattr(kernel_pack, "weights")
                else 0
            )
            flops.append(kernel.get_flops(i))
        assert kernel.get_total_flops() == sum(
            flops
        ), f"Kernel {kernel} has {kernel.get_total_flops()} flops, but {sum(flops)} flops are found in the model"

        self.flops, self.params = self.profile(
            batch_size, force_update=True, seq_len=seq_len
        )

        if compile:
            torch._dynamo.reset()

        for i, (placeholder, kernel_pack) in enumerate(zip(placeholders, kernel_packs)):
            placeholder.reload(kernel_pack, compile)
            placeholder.referred_layer = None

        return "LOAD_SUCCESS"

    def remove_thop_hooks(self):
        for m in self.modules():
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks = OrderedDict()
            if hasattr(m, "_backward_hooks"):
                m._backward_hooks = OrderedDict()
            if hasattr(m, "_buffers"):
                for key in ["total_ops", "total_params"]:
                    if key in m._buffers:
                        m._buffers.pop(key)

    def profile(
        self,
        batch_size=1,
        force_update=False,
        not_count_placeholder=False,
        seq_len=None,
    ) -> Tuple[int, int]:
        if not (self.flops == 0 and self.params == 0) and not force_update:
            return self.flops, self.params

        # Get statistics (count with batch size = 1)
        def count_placeholder_non_zero(m: Placeholder, x, y):
            if m.kernel:
                m.total_ops += torch.DoubleTensor([m.flops])
            else:
                m.total_ops += m.referred_layer.total_ops

        def count_placeholder_zero(m: Placeholder, x, y):
            pass

        count_placeholder = (
            count_placeholder_zero
            if not_count_placeholder
            else count_placeholder_non_zero
        )
        sample_input = torch.ones(
            (batch_size, *self.sample_input_shape(seq_len))
        ).cuda()
        if seq_len:
            sample_input = sample_input.long()
        flops, params = thop.profile(
            self.float(),
            inputs=(sample_input,),
            verbose=False,
            report_missing=False,
            custom_ops={
                Placeholder: count_placeholder,
                LinearPlaceholder: count_placeholder,
                ConvPlaceholder: count_placeholder,
                ViTLinearPlaceholder: count_placeholder,
                ConvNeXtLinearPlaceholder: count_placeholder,
            },
        )
        flops = flops // batch_size
        return int(flops), int(params)

    def sample_input_shape(self, seq_len=None):
        assert False, "Not implemented"

    def sampler_parameters(self, seq_len=None):
        assert False, "Not implemented"

    def initialize_weights(self):
        self.apply(KAS.init_weights)

    def forward(self, x):
        return x
