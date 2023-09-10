"""
Not finished test. 
"""

import os, sys
import logging
from argparse import Namespace

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import log, parser, dataset, models, trainer

from KAS import Assembler, Assembled, Path, Sampler


class Impl:
    def __init__(self, assembler: Assembler) -> None:
        self.assembler = assembler

    def Conv2d_simple(self) -> Assembled:
        N, H, W, k, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1, w_k_2],
        )

    def Conv2d_dilation(self) -> Assembled:
        N, H, W, k_1, k_2, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "k_2", "s", "C_in", "C_out"
        )
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k_1, k_1)

        main_H, windows_H = self.assembler.create_unfold(in_H, k_1 * k_1)
        main_W, windows_W = self.assembler.create_unfold(in_W, k_1 * k_1)

        windows_H_strided = self.assembler.create_stride(windows_H, k_1)
        windows_W_strided = self.assembler.create_stride(windows_W, k_1)

        shared_k_1 = self.assembler.create_share(windows_H_strided, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W_strided, w_k_2)
        shared_C_in = self.assembler.create_share(in_C, w_in_C)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        shared_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1, w_k_2],
        )

    def Conv2d_group(self) -> Assembled:
        N, H, W, k_1, k_2, s, C_in, C_out = self.assembler.get_sizes(
            "N", "H", "W", "k_1", "k_2", "s", "C_in", "C_out"
        )
        k = k_1
        g = s * s  # 4
        (
            in_N,
            in_H,
            in_W,
            in_C,
            out_C,
            w_in_C,
            w_k_1,
            w_k_2,
        ) = self.assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in / g, k, k)

        in_G, in_C_group = self.assembler.create_split(in_C, g)

        main_H, windows_H = self.assembler.create_unfold(in_H, k)
        main_W, windows_W = self.assembler.create_unfold(in_W, k)

        shared_k_1 = self.assembler.create_share(windows_H, w_k_1)
        shared_k_2 = self.assembler.create_share(windows_W, w_k_2)
        shared_C_in = self.assembler.create_share(in_C_group, w_in_C)
        final_C_in = self.assembler.create_merge(in_G, shared_C_in)

        in_N.output(0)
        out_C.output(1)
        main_H.output(2)
        main_W.output(3)
        shared_k_1.sum()
        shared_k_2.sum()
        final_C_in.sum()

        return self.assembler.assemble(
            "conv",
            "in_0 * in_1",
            [in_N, in_C, in_H, in_W],
            [out_C, w_in_C, w_k_1, w_k_2],
        )


def train(
    args: Namespace,
    name: str,
    train_dataloader: dataset.FuncDataloader,
    val_dataloader: dataset.FuncDataloader,
) -> None:

    model, sampler = models.get_model(args, return_sampler=True)
    impl = Impl(sampler.create_assembler())
    assert hasattr(impl, name), f"{name} is not a valid kernel"
    kernel = getattr(impl, name)()

    logging.debug(f"Assembled path: {kernel.convert_to_path(sampler)}")
    if sampler.visit(kernel.convert_to_path(sampler)) is None:
        path = Path(kernel.convert_to_path(sampler))
        logging.warning(f"Path {path} is not valid, testing...")
        for subpath in path.hierarchy:
            if sampler.visit(subpath) is None:
                logging.warning(f"Subpath {subpath} is not valid")
                break

    model.load_kernel(
        sampler, kernel, name=name, compile=args.compile, batch_size=args.batch_size
    )
    flops, params = model.profile(args.batch_size)
    logging.debug(
        f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
    )

    logging.info("Evaluating on real dataset ...")
    accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
    print(f"Evaluation result: {flops} {params} {accuracy}")

    print(f"[Passed\tOK] {name}")


def test_semantic_conv2d() -> None:
    args = parser.arg_parse()

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    train(
        args,
        "Conv2d_simple",
        train_dataloader,
        val_dataloader,
    )
    train(
        args,
        "Conv2d_dilation",
        train_dataloader,
        val_dataloader,
    )
    # train(
    #     args,
    #     "Conv2d_group",
    #     train_dataloader,
    #     val_dataloader,
    # )


if __name__ == "__main__":
    log.setup(level=logging.WARNING)
    test_semantic_conv2d()
