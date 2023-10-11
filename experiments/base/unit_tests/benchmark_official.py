"""
Not finished test. 
"""

import os, sys, json
import logging
from argparse import Namespace
from torch import nn

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import log, parser, dataset, models, trainer

from KAS import Path


class group_conv_oas(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.c_in = layer.in_channels
        self.c_out = layer.out_channels
        self.k = layer.kernel_size[0]
        self.p = layer.padding[0]
        self.g = 32
        self.r = 2

        self.conv = nn.Conv2d(
            self.c_in * self.k * self.k,
            self.c_out // self.r,
            kernel_size=1,
            padding=0,
            groups=self.g,
            bias=False,
        )
        self.fc = nn.Conv2d(
            self.c_out // self.r, self.c_out, kernel_size=1, padding=0, bias=False
        )

    def forward(self, x):
        # (N, C_in, H, W)
        n, c_in, h, w = x.size()
        assert c_in == self.c_in
        t_1 = nn.functional.unfold(x, (self.k, self.k), padding=(self.p, self.p)).view(
            n, c_in * self.k * self.k, h, w
        )
        # (N, C_in * k * k, h, w)
        t_2 = self.conv(t_1)
        # (N, C_out // r, h, w)
        t_3 = self.fc(t_2)
        # (N, C_out, h, w)
        return t_3


def train(
    args: Namespace,
    train_dataloader: dataset.FuncDataloader,
    val_dataloader: dataset.FuncDataloader,
    test_run: bool,
) -> None:

    model, sampler = models.get_model(args, return_sampler=True)
    # logging.info(f"model verbose: {model}")
    placeholders = sampler._extract_placeholders(model)
    # for placeholder in placeholders[1:]:
    #     placeholder.referred_layer = group_conv_oas(placeholder.referred_layer).cuda()

    flops, params = model.profile(args.batch_size)
    logging.info(
        f"Loaded model has {flops / 1e9}G FLOPs per batch and {params / 1e6}M parameters in total."
    )

    if test_run:
        logging.info("Evaluating on real dataset ...")
        model = model.cuda()
        accuracy = max(
            trainer.train(
                model, train_dataloader, val_dataloader, args, init_weight=False
            )
        )


def test_semantic_conv2d(test_kernels, test_run) -> None:
    args = parser.arg_parse()

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    result = train(args, train_dataloader, val_dataloader, test_run)


if __name__ == "__main__":
    log.setup(level=logging.INFO)

    test_kernels = [
        # "Conv2d_simple",
        # "Conv2d_dilation",
        # "Conv2d_group",
        # "Conv2d_FC",
        # "Conv2d_group_oas",
        # "Conv2d_pool",
        # "Conv2d_pool1d",
        # "Conv1d_shift1d",
        # "Shift2d",
    ]
    test_run = True

    test_semantic_conv2d(test_kernels, test_run)

# conv2d 0.7666529605263158
# with fc 0.7532894736842105
# 0.24380672G flops
