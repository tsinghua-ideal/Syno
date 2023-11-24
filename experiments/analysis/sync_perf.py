import argparse
import json
import os, shutil

if __name__ == "__main__":
    # Get Path
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--dirs", type=str, nargs="+", default=[])
    parser.add_argument("--destinations", type=str, nargs="+", default=[])
    args = parser.parse_args()

    path2perf = {}
    # collect
    for dir in args.dirs:
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            if "rej" in kernel_dir:
                continue
            if "uncanon" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            assert os.path.exists(os.path.join(kernel_dir, "perf")), os.path.join(
                kernel_dir, "perf"
            )
            path2perf[meta["path"]] = os.path.join(kernel_dir, "perf")

    json.dump(path2perf, open("path2perf.json", "w"))

    for dest_dir in args.destinations:
        for kernel_fmt in os.listdir(dest_dir):
            kernel_dir = os.path.join(dest_dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if os.path.exists(os.path.join(kernel_dir, "perf")):
                shutil.rmtree(os.path.join(kernel_dir, "perf"))

            if meta["path"] not in path2perf:
                continue

            assert os.path.exists(path2perf[meta["path"]]), path2perf[meta["path"]]
            shutil.copytree(
                path2perf[meta["path"]],
                os.path.join(kernel_dir, "perf"),
                dirs_exist_ok=True,
            )
