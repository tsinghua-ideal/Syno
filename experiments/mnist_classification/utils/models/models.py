from torch import nn, Tensor, Size
from thop import profile
from typing import List, Callable

from KAS import KernelPack, Sampler


class ModelBackup:
    def __init__(self, model: Callable, sample_input: Tensor, device='cuda:0') -> None:
        self._model_builder = model
        self._sample_input = sample_input.to(device)

        self._model = self._model_builder().to(device)
        macs, params = profile(self._model, inputs=(
            self._sample_input, ), verbose=False)
        print(
            f"Referenced model has {round(macs / 1e9, 3)}G MACs and {round(params / 1e6, 3)}M parameters ")
        self.base_macs = macs
        print(f"Base MACs is {self.base_macs}")

    def create_instance(self) -> nn.Module:
        self._model._initialize_weight()
        return self._model

    def restore_model_params(self, model, pack: List[KernelPack]):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        assert len(pack) > 0, "Not detected any placeholders! "
        assert isinstance(pack[0], KernelPack
                          ), f"elements in pack are not valid! {type(pack[0])}"
        Sampler.replace(model, pack)
        return model


def mapping_func_conv(in_size: Size, out_size: Size):
    N, C1, H, W = in_size
    N2, C2, H2, W2 = out_size
    assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
    assert H2 == H and W2 == W, f"Not using same padding. {in_size}->{out_size}"
    mapping = {"N": N, "C_in": C1, "C_out": C2, "H": H, "W": W}
    return mapping


def mapping_func_gray_conv(in_size: Size, out_size: Size):
    N, H, W = in_size
    N2, C2, H2, W2 = out_size
    assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
    assert H2 == H and W2 == W, f"Not using same padding. {in_size}->{out_size}"
    mapping = {"N": N, "C_out": C2, "H": H, "W": W}
    return mapping


def mapping_func_linear(in_size: Size, out_size: Size):
    N, C1 = in_size
    N2, C2 = out_size
    assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
    mapping = {"N": N, "C_in": C1, "C_out": C2}
    return mapping


class KASModule(nn.Module):
    """
    A wrapper of nn.Module for KAS. Now implemented weight initialization. (Maybe more in the future)

    To use it, redefine ``__init__`` and ``forward``. 
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def _initialize_weight(self):
        self.apply(self.init_weights)

    def forward(self, image: Tensor) -> Tensor:
        return image
