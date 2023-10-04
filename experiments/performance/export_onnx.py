import torch
from torch.onnx import TrainingMode, OperatorExportTypes

from KAS.Placeholder import enable_export_for_placeholders, ExportType

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser


if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    model = models.get_model(args, return_sampler=False)
    model.eval()
    enable_export_for_placeholders(model, ExportType.ONNX)
    shape = (args.batch_size, *model.sample_input_shape())
    model_module = f'model_onnx/{args.model}.py'
    os.makedirs(os.path.dirname(model_module), exist_ok=True)
    torch.onnx.export(
        model=model,
        args=torch.randn(shape, device='cuda'),
        f=model_module,
        verbose=True,
        training=TrainingMode.EVAL,
        do_constant_folding=True,
        export_params=False,
        operator_export_type=OperatorExportTypes.ONNX,
    )
