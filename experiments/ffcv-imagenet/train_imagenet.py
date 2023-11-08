import torch as ch

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import os, sys
from fastargs import get_current_config

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import ImageNetTrainer, make_config

if __name__ == "__main__":
    make_config()

    config = get_current_config()
    model = getattr(models, config["model.arch"])(pretrained=config["model.pretrained"])
    folder = config["logging.folder"]

    ImageNetTrainer.launch_from_args(model, folder)
