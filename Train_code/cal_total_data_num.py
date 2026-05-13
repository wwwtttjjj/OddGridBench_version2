import json
from pathlib import Path

DATA_DIR = Path("./")

jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))

total = 0

for file_path in jsonl_files:
    count = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                print(f"[无效 JSON] {file_path.name}")

    total += count
    print(f"{file_path.name}: {count}")

print("-" * 40)
print(f"total: {total}")