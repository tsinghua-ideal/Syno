import torch
from torch import fx
import numpy as np
import tvm
from tvm.relax.frontend.torch import from_fx

from KAS import Sampler
from KAS.Placeholder import enable_export_for_placeholders, ExportType

from common import get_specialized_model_name

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser


def trace(model: torch.nn.Module, input_info) -> tvm.IRModule:
    model.eval()
    # Use FX tracer to trace the PyTorch model.
    graph_module = fx.symbolic_trace(model)
    # Use the importer to import the PyTorch model to Relax.
    mod: tvm.IRModule = from_fx(graph_module, input_info)
    return mod

def write_to_file(args, model: torch.nn.Module, mod: tvm.IRModule, shape, vanilla: bool = False) -> None:
    specialized_model_name = get_specialized_model_name(args.model, args.batch_size, vanilla=vanilla)
    model_module = f'model_relax/{specialized_model_name}.py'
    os.makedirs(os.path.dirname(model_module), exist_ok=True)
    with open(model_module, 'w') as f:
        f.writelines([
            "import tvm\n",
            "from tvm.script import ir as I\n",
            "from tvm.script import tir as T\n",
            "from tvm.script import relax as R\n",
            "\n",
            "INPUT_SHAPE = ", str(shape), "\n",
            "\n"
            "ALL_MAPPINGS = ", str(Sampler._extract_all_mappings(model)), "\n",
            "\n",
            mod.script(show_meta=True),
            "\n",
        ])

if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    model = models.get_model(args, return_sampler=False)
    enable_export_for_placeholders(model, ExportType.RELAX)

    shape = (args.batch_size, *model.sample_input_shape())
    input_info = [(shape, "float32")]

    mod = trace(model, input_info)
    # Print out the imported model.
    print("KAS model:")
    mod.show()
    write_to_file(args, model, mod, shape, vanilla=False)

    vanilla_model = models.get_vanilla_model(args)
    vanilla_mod = trace(vanilla_model, input_info)
    print("Vanilla model:")
    vanilla_mod.show()
    write_to_file(args, vanilla_model, vanilla_mod, shape, vanilla=True)
