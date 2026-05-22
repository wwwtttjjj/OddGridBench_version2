#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def resolve_images(images, base_dir: Path):
    resolved = []
    for image in images or []:
        path = Path(image)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved.append(str(path))
    return resolved


def convert_file(input_path: Path, output_path: Path, limit: int, image_base_dir: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if limit >= 0 and count >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            problem = item["problem"]
            answer = item.get("answer", "")
            out = {
                "messages": [{"role": "user", "content": problem}],
                "images": resolve_images(item.get("images", []), image_base_dir),
                "solution": answer,
                "data_type": item.get("data_type", ""),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert EasyR1 OddGrid JSONL to ms-swift GRPO JSONL.")
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--val-input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=16, help="Use -1 for all samples.")
    parser.add_argument("--val-limit", type=int, default=8, help="Use -1 for all samples.")
    parser.add_argument("--image-base-dir", required=True, type=Path)
    args = parser.parse_args()

    train_count = convert_file(args.train_input, args.train_output, args.train_limit, args.image_base_dir)
    val_count = convert_file(args.val_input, args.val_output, args.val_limit, args.image_base_dir)
    print(f"Wrote {train_count} train samples to {args.train_output}")
    print(f"Wrote {val_count} val samples to {args.val_output}")


if __name__ == "__main__":
    main()
