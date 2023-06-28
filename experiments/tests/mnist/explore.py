import os
import sys
import torch

# KAS
from KAS import Sampler, Explorer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import KASFC, ModelBackup
from utils.parser import arg_parse
from utils.config import parameters

args = arg_parse()
assert 'mnist' in args.dataset
training_params, sampler_params, extra_args = parameters(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ = ModelBackup(KASFC, torch.randn(
    extra_args["sample_input_shape"]), device)
sampler = Sampler(
    net=model_.create_instance(),
    **sampler_params
)

explorer = Explorer(sampler)
explorer.interactive()
