import logging
import os
import sys
import torch

from KAS import KernelLoader
from KAS.Placeholder import build_placeholder_mappings, remove_unsatisfied_placeholders

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from base import models

def get_model(model_name: str, batch_size: int, num_classes: int, input_size: tuple[int, int, int], kernels_dir: str | None) -> torch.nn.Module:
    assert model_name.startswith("torchvision/")
    model_args = {
        "name": model_name[len("torchvision/"):],
        "num_classes": num_classes,
        "input_size": input_size,
    }

    if kernels_dir is None:
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

        logging.info(f"Replacing kernel with {kernels_dir} ...")
        assert os.path.isdir(kernels_dir)
        logging.info(f"Loading from directory ...")
        kernel_directory = os.path.join(
            kernels_dir, "kernel_scheduler_dir"
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

    return model
