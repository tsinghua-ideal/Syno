import os
import shutil
import sys
import logging
from tqdm import tqdm

from KAS import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from reevaluator import collect_kernels

if __name__ == '__main__':
    log.setup(logging.WARNING)

    # Arguments
    args = parser.arg_parse()

    kernels = collect_kernels(args)
    logging.info(f'Rerunning codegen for {kernels}')

    model, sampler = models.get_model(args, return_sampler=True)

    for kernel_save_dir, path in tqdm(kernels, desc="Regenerating kernels"):
        node = sampler.visit(Path.deserialize(path))
        shutil.copytree(
            sampler.realize(model, node).get_directory(),
            os.path.join(kernel_save_dir, "kernel_scheduler_dir"),
            dirs_exist_ok=True,
        )
