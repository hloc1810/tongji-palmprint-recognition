"""Render selected manually inspected cases; never modifies metrics or extraction.

Run after scripts/run_roi_comparison.py, from any working directory.
These are example annotations, not an exhaustive landmark ground truth.
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palmprint.data.dataset import read_gray
from palmprint.data.tongji_roi import draw_overlay

WS = Path(__file__).resolve().parents[1]
OUT = WS / "outputs/roi_comparison"
REVIEWS = [
    ("session1/05119", "plausible", "Correct gap pair; visually close to Official ROI"),
    ("session1/02578", "wrong_gap_selection", "Finger-side concavity replaces a real web"),
    ("session2/02574", "wrong_gap_selection", "Wrong gap triple; excessive rotation"),
    ("session2/02575", "wrong_gap_selection", "Wrong gap triple; excessive rotation"),
    ("session2/02578", "wrong_gap_selection", "Wrong gap triple; recognition also fails"),
]


def main():
    audits = {a["sample_uid"]: a for a in json.loads((OUT / "landmarks.json").read_text())}
    with (OUT / "visual_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_uid", "visual_verdict", "note"])
        writer.writerows(REVIEWS)
    fig, axes = plt.subplots(len(REVIEWS), 3, figsize=(11, 3 * len(REVIEWS)))
    for index, (uid, verdict, note) in enumerate(REVIEWS):
        session, stem = uid.split("/")
        image = cv2.imread(str(WS / "data/Original Images" / session / f"{stem}.tiff"))
        audit = audits[uid]
        overlay = draw_overlay(image, audit["landmarks"])
        for point in audit["landmarks"]["gaps"]:
            cv2.circle(overlay, tuple(np.int32(point)), 8, (255, 0, 255), 2)
        axes[index, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[index, 0].set_title(f"{uid}\n{verdict}", fontsize=10)
        axes[index, 1].imshow(read_gray(WS / "data/raw" / session / f"{stem}.bmp"),
                              cmap="gray", vmin=0, vmax=255)
        axes[index, 1].set_title("Official ROI", fontsize=10)
        axes[index, 2].imshow(read_gray(OUT / "custom_roi" / session / f"{stem}.png"),
                              cmap="gray", vmin=0, vmax=255)
        axes[index, 2].set_title("Custom ROI", fontsize=10)
        for ax in axes[index]:
            ax.axis("off")
    fig.suptitle("Selected visual review: one plausible crop and four confirmed gap errors\n"
                 "Magenta circles = chosen gap candidates; examples only, no metric filtering",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "visual_review_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
