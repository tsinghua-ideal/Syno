import logging
import json, os
from KAS import Path

from base import log, models, parser, dataset, trainer


if __name__ == "__main__":
    log.setup()

    args = parser.arg_parse()
    lower_bound: float = args.kas_evaluate_lower_bound
    
    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)
    logging.info("Preparing model ...")
    model, sampler = models.get_model(args, return_sampler=True)
    
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
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            if "fp32_accuracy" in meta:
                continue

            kernel_hash = kernel_dir.split("_")[1]

            if meta["accuracy"] > lower_bound:
                logging.info(f"Found {meta['path']} with accuracy {meta['accuracy']}, evaluating with fp32......")
                path = meta["path"]
                node = sampler.visit(Path.deserialize(path)).to_node()
                if node is None:
                    logging.info(f"{meta['path']} is not in the search space, skipping......")
                    continue
                kernel_loader = sampler.realize(model, node)
                model.load_kernel(
                    kernel_loader, 
                    compile=args.compile, 
                    batch_size=args.batch_size, 
                    seq_len=args.gpt_seq_len,
                )
                accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args, use_bf16=False))
                logging.info(f"Accuracy of {path} with fp32 is {accuracy}")
                meta["fp32_accuracy"] = accuracy
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
