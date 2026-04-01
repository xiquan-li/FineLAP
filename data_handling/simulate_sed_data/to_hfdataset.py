import argparse
import json
import os
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Audio, Image
from tqdm import tqdm

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL parse error at line {line_no}: {e}") from e
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="目录，包含 audio/ timeline/ sed_metadata.jsonl")
    ap.add_argument("--repo_id", required=True, help='例如 "yourname/fsd50k_sed_dev"')
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_shard_size", default="2GB")
    args = ap.parse_args()

    metadata_path = os.path.join(args.data_root, "sed_metadata.jsonl")
    audio_dir = os.path.join(args.data_root, "audio")
    timeline_dir = os.path.join(args.data_root, "timeline")

    rows = read_jsonl(metadata_path)

    examples = []
    missing = 0

    for r in tqdm(rows):
        audio_path = r.get("audio_path")
        timeline_path = r.get("timeline_path")

        if not audio_path:
            audio_id = r.get("audio_id", "")
            audio_path = os.path.join(audio_dir, f"{audio_id}.wav")

        if not timeline_path:
            audio_id = r.get("audio_id", "")
            timeline_path = os.path.join(timeline_dir, f"{audio_id}_timeline.png")

        if not os.path.exists(audio_path) or not os.path.exists(timeline_path):
            missing += 1
            continue

        phrases = []
        for p in r.get("phrases", []):
            segs = []
            for seg in p.get("segments", []):
                if not isinstance(seg, list) or len(seg) != 2:
                    continue
                segs.append({"start": float(seg[0]), "end": float(seg[1])})
            phrases.append(
                {
                    "phrase": str(p.get("phrase", "")),
                    "segments": segs,
                    "original_clip_id": int(p.get("original_clip_id", -1)),
                }
            )

        examples.append(
            {
                "audio_id": str(r.get("audio_id", "")),
                "caption": str(r.get("caption", "")),
                "audio": audio_path,
                "timeline": timeline_path,
                "phrases": phrases,
            }
        )

    if not examples:
        raise RuntimeError("No valid examples found (all missing files?)")

    ds = Dataset.from_list(examples)
    ds = ds.cast_column("audio", Audio())
    ds = ds.cast_column("timeline", Image())

    dsd = DatasetDict({args.split: ds})

    dsd.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size=args.max_shard_size,
        embed_external_files=True,
    )

    print(f"Done. uploaded {len(examples)} samples. skipped_missing={missing}")


if __name__ == "__main__":
    main()
    
"""
python data_handling/simulate_sed_data/to_hfdataset.py \
  --data_root "../../data/FSD50K/clips/dev_simulated_sed_data" \
  --repo_id "AndreasXi/FineLAP-100k" \
  --private
"""
