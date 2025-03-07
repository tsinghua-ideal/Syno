import os, sys, shutil
from pathlib import Path
from tqdm import tqdm

good_kernel_path = sys.argv[1]

main_prefix = Path("/cephfs/suzhengyuan/KAS-next/experiments/results")
from_prefix = Path("/cephfs/shared/Syno/results")
to_prefix = Path("/cephfs/suzhengyuan/kas-a100-benchmark-results")

for kernel_fmt in os.listdir(from_prefix / good_kernel_path):
    print(kernel_fmt)
    from_kernel_dir = from_prefix / good_kernel_path / kernel_fmt
    main_kernel_dir = main_prefix / good_kernel_path / kernel_fmt
    to_kernel_dir = to_prefix / good_kernel_path / kernel_fmt
    os.makedirs(to_kernel_dir, mode=644, exist_ok=True)
    shutil.copy(main_kernel_dir / "graph.dot", to_kernel_dir / "graph.dot")
    shutil.copy(main_kernel_dir / "loop.txt", to_kernel_dir / "loop.txt")
    shutil.copy(main_kernel_dir / "meta.json", to_kernel_dir / "meta.json")
    if os.path.exists(main_kernel_dir / "meta_new.json"):
        shutil.copy(main_kernel_dir / "meta_new.json", to_kernel_dir / "meta_new.json")
    shutil.copytree(main_kernel_dir / "kernel_scheduler_dir", to_kernel_dir / "kernel_scheduler_dir", dirs_exist_ok=True)
    if os.path.exists(to_kernel_dir / "perf"):
        os.remove(to_kernel_dir / "perf")
    os.symlink(from_kernel_dir / "perf", to_kernel_dir / "perf", target_is_directory=True)
