import torch
from torch import fx
import numpy as np
import tvm
from tvm.relax.frontend.torch import from_fx

from KAS.Placeholder import enable_onnx_for_placeholders

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser


if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    model = models.get_model(args, return_sampler=False)
    model.eval()
    enable_onnx_for_placeholders(model)
    shape = (args.batch_size, *model.sample_input_shape())
    
    input_info = [(shape, "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Use FX tracer to trace the PyTorch model.
    graph_module = fx.symbolic_trace(model)

    # Use the importer to import the PyTorch model to Relax.
    mod: tvm.IRModule = from_fx(graph_module, input_info)

    # Print out the imported model.
    print(mod.script())

    os.makedirs('model_relax', exist_ok=True)
    with open(f'model_relax/{args.model}.py', 'w') as f:
        f.writelines(["import tvm\n", "from tvm.script import ir as I\n", "from tvm.script import relax as R\n"])
        f.write(mod.script(show_meta=True))
