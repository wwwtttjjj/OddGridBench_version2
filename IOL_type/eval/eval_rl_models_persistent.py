#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

from vllm_infer_dire import run_vllm_http


def main():
    parser = argparse.ArgumentParser(
        description="Run one model across multiple data types in one Python process, so vLLM loads the model once."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--image_type", default="normal")
    parser.add_argument("--data_types", nargs="+", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print(f"[PERSISTENT] Load once and run model: {args.model_name}")
    print(f"[PERSISTENT] Data types: {', '.join(args.data_types)}")
    print("=" * 80)

    for data_type in args.data_types:
        print("-" * 80)
        print(f"[PERSISTENT] Running model={args.model_name}, data_type={data_type}")
        print("-" * 80)
        run_vllm_http(SimpleNamespace(
            model_name=args.model_name,
            image_type=args.image_type,
            data_type=data_type,
        ))

    print("=" * 80)
    print(f"[PERSISTENT] Finished model: {args.model_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
