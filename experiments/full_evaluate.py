import logging
import json, os
from KAS import KernelLoader

from base import log, models, parser, dataset, trainer


if __name__ == "__main__":
    log.setup()

    args = parser.arg_parse()
    lower_bound: float = args.kas_evaluate_lower_bound
    evaluate_type: str = args.evaluate_type

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)
    logging.info("Preparing model ...")
    model = models.get_model(args, return_sampler=False)

    for dir in args.dirs:
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert (
                "graph.dot" in files
                and "loop.txt" in files
                and "meta.json" in files
                and "kernel_scheduler_dir" in files
            )

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if f"{evaluate_type}_accuracy" in meta:
                continue

            if meta["accuracy"] > lower_bound:
                logging.info(
                    f"Found {meta['path']} with accuracy {meta['accuracy']}, evaluating with {evaluate_type}......"
                )
                path = meta["path"]
                kernel_directory = os.path.join(kernel_dir, "kernel_scheduler_dir")
                kernel_loader = KernelLoader.from_directory(kernel_directory)
                kernel_flag = model.load_kernel(
                    kernel_loader,
                    compile=args.compile,
                    batch_size=args.batch_size,
                    seq_len=args.gpt_seq_len,
                )
                assert kernel_flag == "LOAD_SUCCESS", kernel_flag
                accuracy = max(
                    trainer.train(
                        model,
                        train_dataloader,
                        val_dataloader,
                        args,
                        use_bf16_train=False,
                        use_bf16_test=False,
                    )
                )
                logging.info(f"Accuracy of {path} with {evaluate_type} is {accuracy}")
                meta[f"{evaluate_type}_accuracy"] = accuracy
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
