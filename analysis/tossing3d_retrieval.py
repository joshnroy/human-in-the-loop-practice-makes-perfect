"""Plot observed controller outcomes from the scripted physical regression."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class RetrievalPlot:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        records = json.loads(args.input.read_text())
        keys = ("holding", "in_bin", "reachable")
        data = np.array([[int(row[key]) for row in records] for key in keys])
        fig, ax = plt.subplots(figsize=(9, 3.1))
        ax.imshow(data, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(records)), [row["step"] for row in records])
        ax.set_yticks(range(3), ["Holding", "In bin", "Same side"])
        for y in range(3):
            for x in range(len(records)):
                ax.text(
                    x,
                    y,
                    "true" if data[y, x] else "false",
                    ha="center",
                    va="center",
                    color="white" if data[y, x] else "black",
                )
        ax.set_title(
            "Physical regression: hit → bin retrieval → miss → floor retrieval\n"
            "Scripted controllers; one initial reset; seed 125"
        )
        fig.tight_layout()
        fig.savefig(args.output, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    RetrievalPlot.main()
