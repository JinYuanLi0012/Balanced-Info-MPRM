"""
Build BIS-selected training subset from VisualPRM400K annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BIS-selected PRM subset from VisualPRM400K annotations."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("datasets/VisualPRM400K-v1.1-raw/annotations"),
        help="Directory containing *.jsonl annotation files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Smoothing term alpha in BIS(x) = (p_pos*(1-p_pos) + alpha) * R.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("faker_project/BIS/bis25_alpha0_05_combined_prm.jsonl"),
        help=(
            "Path to output jsonl for the selected subset in PRM format "
            "(image + conversations)."
        ),
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.25,
        help="Fraction of rollouts to keep per file (default: 0.25).",
    )
    return parser.parse_args()


def extract_step_scores(record: dict) -> List[float]:
    """Extract list of MC scores from steps_with_score."""
    out: List[float] = []
    for step in record.get("steps_with_score") or []:
        sc = step.get("score")
        if sc is None:
            continue
        try:
            out.append(float(sc))
        except (TypeError, ValueError):
            continue
    return out


def compute_bis(step_scores: List[float], alpha: float) -> float:
    """
    Compute BIS(x) = (p_pos*(1-p_pos) + alpha) * R for a single rollout.

    - Hard labels: y_j = 1 if score_j > 0 else 0
    - p_pos = n_pos / n
    - R = avg_pos_MC if there is at least one positive step,
          otherwise R = 1.0 (all-negative rollout treated as noise-free negative)
    """
    n = len(step_scores)
    if n == 0:
        raise ValueError("compute_bis called with empty step_scores")

    labels = [1 if sc > 0.0 else 0 for sc in step_scores]
    n_pos = sum(labels)
    p_pos = n_pos / n

    if n_pos > 0:
        avg_pos_mc = sum(sc for sc in step_scores if sc > 0.0) / n_pos
        R = avg_pos_mc
    else:
        R = 1.0

    bis = (p_pos * (1.0 - p_pos) + alpha) * R
    return bis


def norm_question(rec: dict) -> str:
    """Return normalized question text, preferring question_orig."""
    q = rec.get("question_orig") or rec.get("question") or ""
    return str(q).strip()


def build_prm_from_rec(rec: dict, thr: float = 0.0) -> dict | None:
    """
    Convert an annotation record into PRM training format:

    {
      "image": "...",
      "conversations": [
        {"from": "human", "value": "Question: ...\\nProcess: step1<prm>\\n\\nstep2<prm>..."},
        {"from": "gpt", "value": [1.0, 0.0, ...]}
      ]
    }

    Entire rollout is discarded (returns None) if:
      - image is missing/empty;
      - steps_with_score is empty;
      - any step has empty text or missing/invalid score.
    """
    img = rec.get("image", "")
    if not img:
        return None

    steps = rec.get("steps_with_score") or []
    if not steps:
        return None

    proc_lines: List[str] = []
    labels: List[float] = []

    for step in steps:
        st = (step.get("step") or "").strip()
        sc = step.get("score")
        if not st or sc is None:
            return None
        try:
            sc_f = float(sc)
        except (TypeError, ValueError):
            return None
        p = 1.0 if sc_f > thr else 0.0
        proc_lines.append(st + "<prm>")
        labels.append(p)

    if not proc_lines:
        return None

    q = norm_question(rec)
    human_val = f"Question: {q}\nProcess: " + "\n\n".join(proc_lines)
    human = {"from": "human", "value": human_val}
    gpt = {"from": "gpt", "value": labels}
    return {"image": img, "conversations": [human, gpt]}


def build_subset(
    annotations_dir: Path, alpha: float, top_ratio: float, output_path: Path
) -> None:
    """
    Execute BIS-based selection over all *.jsonl files and emit PRM-format
    records for the selected rollouts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(annotations_dir.glob("*.jsonl"))
    if not ann_files:
        print(f"[WARN] No jsonl files found in {annotations_dir}")
        return

    total_written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for ann_path in ann_files:
            with ann_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            scored_indices: List[Tuple[int, float]] = []

            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step_scores = extract_step_scores(rec)
                if not step_scores:
                    continue

                try:
                    score = compute_bis(step_scores, alpha=alpha)
                except ValueError:
                    continue
                scored_indices.append((idx, score))

            if not scored_indices:
                print(f"[INFO] No valid rollouts found in {ann_path.name}, skipping.")
                continue

            scored_indices.sort(key=lambda x: x[1], reverse=True)
            m = len(scored_indices)
            top_k = max(1, int(m * top_ratio))
            chosen = sorted(idx for idx, _ in scored_indices[:top_k])

            print(
                f"{ann_path.name}: valid_rollouts={m}, "
                f"top_ratio={top_ratio:.2f}, selected={len(chosen)}"
            )

            for idx in chosen:
                line = lines[idx].strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prm = build_prm_from_rec(rec, thr=0.0)
                if prm is None:
                    continue
                fout.write(json.dumps(prm, ensure_ascii=False) + "\n")
                total_written += 1

    print(
        f"[DONE] BIS PRM subset written to {output_path} "
        f"with total_written={total_written}"
    )


def main() -> None:
    args = parse_args()
    build_subset(
        annotations_dir=args.annotations_dir,
        alpha=args.alpha,
        top_ratio=args.top_ratio,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
