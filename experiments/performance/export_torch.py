import torch
from torch import fx

from common import get_specialized_model_name

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from base import log, models, parser


if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    model = models.get_vanilla_model(args)
    model.eval()
    module = fx.symbolic_trace(model)

    specialized_model_name = get_specialized_model_name(args.model, args.batch_size, vanilla=True)
    save_path = f"model_torch/{specialized_model_name}"
    os.makedirs(save_path, exist_ok=True)
    module.to_folder(save_path, module_name="ExportedModel")

    shape = (args.batch_size, *args.input_size)
    with open(os.path.join(save_path, "module.py"), "a") as f:
        f.write(f"INPUT_SHAPE = {shape}\n")
