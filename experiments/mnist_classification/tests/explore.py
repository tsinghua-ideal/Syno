import os, sys

# KAS
from KAS import Sampler, Explorer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import KASFC
from utils.parser import arg_parse
from utils.config import parameters

args = arg_parse()
training_params, sampler_params, extra_args = parameters(args)
model = KASFC().cuda()
sampler = Sampler(
    net=model,
    **sampler_params
)

explorer = Explorer(sampler)
explorer.interactive()
