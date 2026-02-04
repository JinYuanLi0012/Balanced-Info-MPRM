# 文件: datasets/VisualPRM400K/tools/convert_visualprm400k_to_mmprm.py
import os
import glob
import jsonlines

SRC_DIR = os.path.join("datasets", "VisualPRM400K", "annotations")
DST_SOFT = os.path.join("datasets", "VisualPRM400K", "converted_soft")
DST_HARD = os.path.join("datasets", "VisualPRM400K", "converted_hard")

# 硬标签阈值，可按需调整
HARD_THR = 0.0

def norm_question(rec):
    # 优先使用 question_orig，没有则用 question
    q = rec.get("question_orig") or rec.get("question") or ""
    return q.strip()

def build_process_and_labels(steps, binarize=False, thr=0.5):
    """
    将 steps_with_score 转为:
      - Process 文本（每步末尾追加 <prm>）
      - 标签列表（软: [p1, p2, ...]; 硬: [0/1, ...]）
    会跳过无效步(无 step 或无 score)；确保步数与标签对齐。
    """
    proc_lines = []
    labels = []
    for s in steps or []:
        st = (s.get("step") or "").strip()
        sc = s.get("score")
        if not st or sc is None:
            continue
        if binarize:
            p = 1.0 if float(sc) > thr else 0.0
        else:
            p = float(sc)
        proc_lines.append(f"{st}<prm>")
        labels.append(p)

    if not proc_lines or not labels:
        return None, None
    return "\n\n".join(proc_lines), labels

def convert_record(rec, binarize=False, thr=0.5):
    """
    输出为本项目所需格式:
    {
      "image": "VisualPRM400K-v1.1-Raw/.../xxx.png",
      "conversations": [
        {"from": "human", "value": "Question: ...\nProcess: ...<prm>\n\n...<prm>"},
        {"from": "gpt",   "value": [p1, p2, ...]}   # 软: 概率; 硬: 0/1
      ]
    }
    """
    img = rec.get("image", "")
    if not img:
        return None

    q = norm_question(rec)
    steps = rec.get("steps_with_score")
    process_text, labels = build_process_and_labels(steps, binarize=binarize, thr=thr)
    if process_text is None:
        return None

    human = {"from": "human", "value": f"Question: {q}\nProcess: {process_text}"}
    gpt = {"from": "gpt", "value": labels}
    return {"image": img, "conversations": [human, gpt]}

def ensure_dirs():
    os.makedirs(DST_SOFT, exist_ok=True)
    os.makedirs(DST_HARD, exist_ok=True)

def main():
    ensure_dirs()
    src_files = glob.glob(os.path.join(SRC_DIR, "*.jsonl"))
    if not src_files:
        print(f"[WARN] No jsonl files found in {SRC_DIR}")
        return

    for src in src_files:
        base = os.path.splitext(os.path.basename(src))[0]
        dst_soft = os.path.join(DST_SOFT, f"{base}_prm.jsonl")
        dst_hard = os.path.join(DST_HARD, f"{base}_prm.jsonl")

        kept_soft = kept_hard = total = 0

        with jsonlines.open(src, "r") as reader, \
             jsonlines.open(dst_soft, "w") as w_soft, \
             jsonlines.open(dst_hard, "w") as w_hard:

            for rec in reader:
                total += 1
                # 软标签
                out_soft = convert_record(rec, binarize=False)
                if out_soft is not None:
                    w_soft.write(out_soft)
                    kept_soft += 1
                # 硬标签
                out_hard = convert_record(rec, binarize=True, thr=HARD_THR)
                if out_hard is not None:
                    w_hard.write(out_hard)
                    kept_hard += 1

        print(f"{base}: total={total}, soft_kept={kept_soft} -> {dst_soft}")
        print(f"{base}: total={total}, hard_kept={kept_hard} -> {dst_hard}")

if __name__ == "__main__":
    main()