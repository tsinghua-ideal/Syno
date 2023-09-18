"""
Reference: https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
"""

import resource


def get_memory() -> int:
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory  # KiB


def memory_limit(ratio: float = 1.0) -> None:
    """Limit max memory usage to half."""
    assert 0.0 < ratio <= 1.0, f"ratio {ratio} is not valid! It should be in (0, 1]. "
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * ratio), hard))
