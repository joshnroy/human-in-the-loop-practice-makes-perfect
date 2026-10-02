"""Fail closed unless the current cgroup enforces sixGiB RAM and zero swap."""

import json
from pathlib import Path


def main():
    relative = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    constraints = []
    current = group
    while current != Path("/sys/fs"):
        row = {"path": str(current)}
        for name in ("memory.max", "memory.swap.max"):
            path = current / name
            if path.exists():
                row[name] = path.read_text().strip()
        constraints.append(row)
        current = current.parent
    ram = [int(r["memory.max"]) for r in constraints if r.get("memory.max", "max") != "max"]
    swap = [
        int(r["memory.swap.max"]) for r in constraints if r.get("memory.swap.max", "max") != "max"
    ]
    if not ram or min(ram) > 6 * 1024**3 or not swap or min(swap) != 0:
        raise RuntimeError(f"Missing required MemoryMax6G/MemorySwapMax0: {constraints}")
    print(
        json.dumps({"memory_max": min(ram), "memory_swap_max": min(swap), "cgroup": str(group)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
