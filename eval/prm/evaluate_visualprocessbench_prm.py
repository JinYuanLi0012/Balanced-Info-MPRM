import argparse
import itertools
import json
import math
import os
import random
import re
import time
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

# optional AUC support; will be skipped if sklearn not installed
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def collate_fn(batches):
    pixel_values = batches[0]['pixel_values']
    prompts = batches[0]['prompts']
    steps_lens = batches[0]['steps_lens']
    data_item = batches[0]['data_item']
    return pixel_values, prompts, steps_lens, data_item


def _clean_question(text: str) -> str:
    text = re.sub(r"<image\d+>", "", text)
    return text.strip()


def _build_prompt(question: str, steps: List[str]) -> Tuple[str, int]:
    solution = '<prm>'.join([s.strip() for s in steps]) + '<prm>' if steps else ''
    prompt = f"Question: {question}\nProcess: {solution}"
    return prompt, len(steps)


def _tile_images(image_paths: List[str], grid_max_cols: int = 3) -> Image.Image:
    images: List[Image.Image] = [Image.open(p).convert('RGB') for p in image_paths]
    if len(images) == 1:
        return images[0]

    cols = min(grid_max_cols, len(images))
    rows = (len(images) + cols - 1) // cols

    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    canvas = Image.new('RGB', (cols * max_w, rows * max_h), (255, 255, 255))
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        # center paste in its cell
        x = c * max_w + (max_w - img.width) // 2
        y = r * max_h + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class VisualProcessBenchPRMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str,
        annotation_path: str,
        input_size: int = 224,
        dynamic_image_size: bool = False,
        use_thumbnail: bool = False,
        max_num: int = 6,
        grid_max_cols: int = 3,
    ):
        self.image_root = image_root
        self.annotation_path = annotation_path
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.grid_max_cols = grid_max_cols
        self.transform = build_transform(is_train=False, input_size=input_size)

        # load jsonl（逐行解析，避免额外依赖）
        self.data: List[Dict[str, Any]] = []
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def _resolve_paths(self, rel_list: List[str]) -> List[str]:
        resolved = []
        for p in rel_list:
            if os.path.isabs(p):
                resolved.append(p)
            else:
                resolved.append(os.path.normpath(os.path.join(self.image_root, p)))
        return resolved

    def __getitem__(self, idx):
        rec = self.data[idx]

        img_list = rec.get('image') or rec.get('images') or []
        assert isinstance(img_list, list) and len(img_list) > 0, f"invalid image field at idx {idx}"
        paths = self._resolve_paths(img_list)
        image = _tile_images(paths, grid_max_cols=self.grid_max_cols)

        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
        else:
            images = [image]
        pixel_values = [self.transform(im) for im in images]
        pixel_values = torch.stack(pixel_values)

        question_raw = rec.get('question', '')
        question = _clean_question(question_raw)

        steps: List[str] = rec.get('response', {}).get('steps', [])
        process_correctness: List[int] = rec.get('response', {}).get('process_correctness', [])
        assert len(steps) == len(process_correctness), f"steps and labels length mismatch at idx {idx}"

        prompt, steps_len = _build_prompt(question, steps)

        data_item = {
            'data_source': rec.get('data_source', 'UNKNOWN'),
            'labels_per_step': process_correctness,  # 1 (pos), -1 (neg), 0 (neutral)
            'meta': {
                'image_paths': paths,
                'question': question,
            },
        }

        return {
            'pixel_values': pixel_values,  # [num_patches, 3, H, W]
            'prompts': [prompt],
            'steps_lens': [steps_len],
            'data_item': data_item,
        }


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def _macro_f1_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    # class +1 as positive
    tp_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp_pos = sum(1 for t, p in zip(y_true, y_pred) if t != 1 and p == 1)
    fn_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p != 1)
    f1_pos = _f1_from_counts(tp_pos, fp_pos, fn_pos)

    # class -1 as positive
    tp_neg = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp_neg = sum(1 for t, p in zip(y_true, y_pred) if t != -1 and p == -1)
    fn_neg = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p != -1)
    f1_neg = _f1_from_counts(tp_neg, fp_neg, fn_neg)

    macro_f1 = (f1_pos + f1_neg) / 2.0
    return {
        'f1_positive': f1_pos,
        'f1_negative': f1_neg,
        'macro_f1': macro_f1,
        'counts': {
            'tp_pos': tp_pos,
            'fp_pos': fp_pos,
            'fn_pos': fn_pos,
            'tp_neg': tp_neg,
            'fp_neg': fp_neg,
            'fn_neg': fn_neg,
        },
    }


def _micro_f1_from_labels(y_true: List[int], y_pred: List[int]) -> float:
    # choose +1 as positive for counting; macro vs micro for binary reduces to same if weighted; but we implement micro via overall TP/FP/FN for positive, symmetric to negative
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p != 1)
    return _f1_from_counts(tp, fp, fn)


def evaluate_model():
    random.seed(args.seed)

    dataset = VisualProcessBenchPRMDataset(
        image_root=args.image_root,
        annotation_path=args.annotation,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        grid_max_cols=args.grid_max_cols,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for idx, (pixel_values, prompts, steps_lens, data_item) in tqdm(enumerate(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        prm_scores_flattened: List[float] = []
        for i in range(0, len(prompts), args.mini_batch_size):
            curr_bs = min(args.mini_batch_size, len(prompts) - i)
            output = model.batch_prm(
                tokenizer=tokenizer,
                pixel_values=torch.cat([pixel_values] * curr_bs, dim=0),
                questions=prompts[i : i + curr_bs],
                num_patches_list=[pixel_values.shape[0]] * curr_bs,
                verbose=False,
            )
            prm_scores_flattened.extend(output.tolist())

        data_item['prm_scores'] = []
        curr_len = 0
        for i in range(len(steps_lens)):
            data_item['prm_scores'].append(
                prm_scores_flattened[curr_len : curr_len + steps_lens[i]]
            )
            curr_len += steps_lens[i]

        outputs.append(data_item)

        if idx % 50 == 0:
            torch.distributed.barrier()

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() != 0:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'visualprocessbench_{time_prefix}.json'
    output_path = os.path.join(args.out_dir, results_file)
    json.dump(merged_outputs, open(output_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'Results saved to {output_path}')

    # Compute metrics
    # 先收集每来源与全局的 (y, s)，过滤中立(0)
    per_source_pairs: Dict[str, Dict[str, list]] = {}
    global_y_all: List[int] = []
    global_s_all: List[float] = []
    for item in merged_outputs:
        src = item.get('data_source', 'UNKNOWN')
        labels: List[int] = item['labels_per_step']
        scores_nested: List[List[float]] = item['prm_scores']
        scores: List[float] = scores_nested[0] if scores_nested else []
        for y, s in zip(labels, scores):
            if y == 0:
                continue
            d = per_source_pairs.setdefault(src, {'y': [], 's': []})
            d['y'].append(1 if y == 1 else -1)
            d['s'].append(float(s))
            global_y_all.append(1 if y == 1 else -1)
            global_s_all.append(float(s))

    # 可选 AUC
    auc = None
    if _HAS_SKLEARN and len(global_y_all) > 0:
        y_auc = [1 if yy == 1 else 0 for yy in global_y_all]
        try:
            auc = roc_auc_score(y_auc, global_s_all)
        except Exception:
            auc = None

    # 自动扫阈值（以“按来源 Macro-F1 的 micro 平均”为优化目标；同时报告池化 Macro-F1 的最优阈值）
    selected_threshold = args.threshold
    auto_info = None
    if args.auto and len(global_s_all) > 0:
        # 候选阈值：分数去重后四舍五入到 1e-4，数量过大则等距降采样至 ~1000
        cands = sorted(set(round(x, 4) for x in global_s_all))
        if len(cands) > 1000:
            step = max(1, len(cands) // 1000)
            cands = cands[::step]

        best_micro = (-1.0, 0.5)   # (score, thr)
        best_pooled = (-1.0, 0.5)  # (score, thr)

        for t in cands:
            # micro over sources（各来源 Macro-F1 用有效步数加权）
            weighted_macro_sum = 0.0
            counted = 0
            for d in per_source_pairs.values():
                if not d['y']:
                    continue
                y_pred_src = [1 if ss >= t else -1 for ss in d['s']]
                m_src = _macro_f1_binary(d['y'], y_pred_src)
                n_src = len(d['y'])
                weighted_macro_sum += m_src['macro_f1'] * n_src
                counted += n_src
            micro = (weighted_macro_sum / counted) if counted > 0 else 0.0
            if micro > best_micro[0]:
                best_micro = (micro, t)

            # 池化后 Macro-F1
            y_pred_global = [1 if ss >= t else -1 for ss in global_s_all]
            overall_macro = _macro_f1_binary(global_y_all, y_pred_global)
            if overall_macro['macro_f1'] > best_pooled[0]:
                best_pooled = (overall_macro['macro_f1'], t)

        selected_threshold = best_micro[1]
        auto_info = {
            'auc': auc,
            'best_threshold_micro_over_sources': {'threshold': best_micro[1], 'score': best_micro[0]},
            'best_threshold_pooled_macro': {'threshold': best_pooled[1], 'score': best_pooled[0]},
        }
        print(f"[auto] candidates: {len(cands)}; AUC: {auc:.4f}" if auc is not None else f"[auto] candidates: {len(cands)}; AUC: N/A")
        print(f"[auto] Best micro over sources = {best_micro[0]:.4f} @ threshold = {best_micro[1]:.4f}")
        print(f"[auto] Best pooled Macro-F1 = {best_pooled[0]:.4f} @ threshold = {best_pooled[1]:.4f}")
        print(f"[auto] Using threshold = {selected_threshold:.4f} for final metrics")

    # 使用 selected_threshold 计算最终各来源与总体指标（保持原有输出格式）
    per_source: Dict[str, Dict[str, Any]] = {}
    global_y_true: List[int] = []
    global_y_pred: List[int] = []

    for item in merged_outputs:
        source = item.get('data_source', 'UNKNOWN')
        labels: List[int] = item['labels_per_step']
        scores_nested: List[List[float]] = item['prm_scores']
        scores: List[float] = scores_nested[0] if scores_nested else []

        pairs = [(y, 1 if s >= selected_threshold else -1) for y, s in zip(labels, scores) if y != 0]
        if not pairs:
            continue
        y_true, y_pred = zip(*pairs)
        y_true = list(y_true)
        y_pred = list(y_pred)

        ms = per_source.setdefault(source, {'y_true': [], 'y_pred': []})
        ms['y_true'].extend(y_true)
        ms['y_pred'].extend(y_pred)

        global_y_true.extend(y_true)
        global_y_pred.extend(y_pred)

    metrics_summary = {'per_source': {}, 'overall': {}}
    for src, d in per_source.items():
        m = _macro_f1_binary(d['y_true'], d['y_pred'])
        metrics_summary['per_source'][src] = m
    
    # 按来源做 micro 平均（用该来源有效步数加权）
    total_steps_over_sources = sum(len(d['y_true']) for d in per_source.values())
    weighted_macro_sum = 0.0
    for src, d in per_source.items():
        n_src = len(d['y_true'])
        if n_src == 0:
            continue
        weighted_macro_sum += metrics_summary['per_source'][src]['macro_f1'] * n_src
    micro_over_sources = (weighted_macro_sum / total_steps_over_sources) if total_steps_over_sources > 0 else 0.0

    # 参考：池化后 Macro-F1（所有来源合并后再算一次）
    overall_macro = _macro_f1_binary(global_y_true, global_y_pred)
    metrics_summary['overall'] = {
        'micro_over_sources': micro_over_sources,
        'macro_f1_pooled': overall_macro['macro_f1'],
        'f1_positive': overall_macro['f1_positive'],
        'f1_negative': overall_macro['f1_negative'],
        'total_steps': len(global_y_true),
        'threshold_used': selected_threshold,
    }
    if auto_info is not None:
        metrics_summary['auto_search'] = auto_info

    metrics_path = os.path.join(args.out_dir, f'metrics_{time_prefix}.json')
    json.dump(metrics_summary, open(metrics_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"Selected threshold: {selected_threshold:.4f} (auto={args.auto})")
    print('Per-source Macro-F1:')
    for src, m in metrics_summary['per_source'].items():
        print(f" - {src}: macro_f1={m['macro_f1']:.4f} (pos={m['f1_positive']:.4f}, neg={m['f1_negative']:.4f})")

    # 论文口径：按来源的 Macro-F1 做 micro 平均
    print(f"Overall (micro over sources): {metrics_summary['overall']['micro_over_sources']:.4f}")

    # 参考口径：池化后 Macro-F1（与之前的 Overall 等价）
    print(
        f"Pooled Macro-F1: {metrics_summary['overall']['macro_f1_pooled']:.4f} "
        f"(pos={metrics_summary['overall']['f1_positive']:.4f}, "
        f"neg={metrics_summary['overall']['f1_negative']:.4f}); "
        f"steps counted: {metrics_summary['overall']['total_steps']}"
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--image-root', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--mini-batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--grid-max-cols', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_model()


