import importlib.util
import logging
import os
import sys
import torch

from KAS import KernelLoader
from KAS.Placeholder import build_placeholder_mappings, remove_unsatisfied_placeholders

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from base import models

# name -> (C_in, C_out, H, k, placeholder_index)
RESNET34_LAYERS = {
    "conv_io64": (64, 64, 56, 3, 0),
    "conv_i64_o128": (64, 128, 28, 3, 6),
    "conv_io128": (128, 128, 28, 3, 7),
    "residual_i64_o128": (64, 128, 28, 1, 8),
    "conv_i128_o256": (128, 256, 14, 3, 15),
    "conv_io256": (256, 256, 14, 3, 16),
    "residual_i128_o256": (128, 256, 14, 1, 17),
    "conv_i256_o512": (256, 512, 7, 3, 28),
    "conv_io512": (512, 512, 7, 3, 29),
    "residual_i256_o512": (256, 512, 7, 1, 30),
}
# In residual layers, we always use k=3 for substituted kernels in KAS, so for fair comparison, we apply that to NAS-PTE as well.
RESNET34_LAYERS_K_SUBSTITUTED = 3

def get_resnet34_layers(layer_name: str, result_dir: str | None, batch_size: int) -> tuple[torch.nn.Module, tuple[int, int, int, int]]:
    C_in, C_out, H, k, placeholder_index = RESNET34_LAYERS[layer_name]
    input_shape = (batch_size, C_in, H, H)

    if result_dir is None:
        return torch.nn.Conv2d(C_in, C_out, (k, k)), input_shape
    else:
        kernel_directory = os.path.join(result_dir, "kernel_scheduler_dir")
        if os.path.isdir(kernel_directory):
            # This is normal Syno-generated kernel
            kernel_loader = KernelLoader.from_directory(kernel_directory)
            kernel_packs = kernel_loader.construct_kernel_packs()
            assert 35 == kernel_loader.get_count_placeholders(), f"This is not a ResNet-34 kernel, got {kernel_loader.get_count_placeholders()} placeholders"
            return kernel_packs[placeholder_index], input_shape
        else:
            # This is NAS-PTE.
            kernel_file = os.path.join(result_dir, "kernels_torch.py")
            kernel_name = kernel_file.split(".py")[0].replace("/", ".").replace("-", "_")
            spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
            assert spec is not None, f"Failed to load kernel from {kernel_file}"
            kernels = importlib.util.module_from_spec(spec)
            sys.modules[kernel_name] = kernels
            spec.loader.exec_module(kernels)
            return kernels.kernel_generated(C_in=C_in, C_out=C_out, H=H, k=RESNET34_LAYERS_K_SUBSTITUTED), input_shape

def get_model(model_name: str, result_dir: str | None, batch_size: int, input_size: tuple[int, int, int], num_classes: int) -> tuple[torch.nn.Module, tuple[int, int, int, int]]:
    if model_name.startswith("resnet34layers/"):
        return get_resnet34_layers(model_name[len("resnet34layers/"):], result_dir, batch_size)
    assert model_name.startswith("torchvision/"), f"Invalid model name {model_name}, only torchvision models are supported"
    model_args = {
        "name": model_name[len("torchvision/"):],
        "num_classes": num_classes,
        "input_size": input_size,
    }

    if result_dir is None:
        model = models.common._get_vanilla_common_model(**model_args)
    else:
        # Replace kernel
        model = models.common.CommonModel(**model_args)
        flops, params = model.profile()
        logging.info(
            f"Base model {model_name} has {flops / 1e9:.5f} GFLOPs (per batch) and {params / 1e6:.2f}M parameters"
        )

        # Build mapping for usages
        sample_input = torch.ones(
            (batch_size, *model.sample_input_shape())
        )
        build_placeholder_mappings(model, sample_input)
        count = remove_unsatisfied_placeholders(model)
        logging.info(f"Recovered {count} unsatisfied placeholders")

        logging.info(f"Replacing kernel with {result_dir} ...")
        assert os.path.isdir(result_dir)
        logging.info(f"Loading from directory ...")
        kernel_directory = os.path.join(
            result_dir, "kernel_scheduler_dir"
        )
        kernel_loader = KernelLoader.from_directory(kernel_directory)
        model.load_kernel(
            kernel_loader,
            batch_size=batch_size,
        )
        flops_replaced, params_replaced = model.profile(
            batch_size=batch_size, force_update=True,
        )
        flops_base, params_base = model.profile(
            batch_size=batch_size,
            force_update=True,
            not_count_placeholder=True,
        )
        logging.info(
            f"Replaced model {model_name} has {flops_replaced / 1e9:.5f}G FLOPs and {params_replaced / 1e6:.2f}M parameters"
        )
        logging.info(
            f"Placeholder flops change {flops - flops_base:.2f} -> {flops_replaced - flops_base:.2f}"
        )
        logging.info(
            f"Placeholder params change {params - params_base:.2f} -> {params_replaced - params_base:.2f}"
        )

    return model, (batch_size, *input_size)
