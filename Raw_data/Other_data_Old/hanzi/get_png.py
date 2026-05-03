import json
import random
import sys
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont


# ===============================
# ✅ 字体文件（用户级路径即可）
# ===============================

FONT_DIR = Path("../SimplifiedChinese")


# 预加载字体缓存（按 font_size）
FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}


# ===============================
# Pair-level style sampler（保持你原逻辑）
# ===============================

def sample_pair_style(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {
        "font_size": rng.randint(42, 52),
        "fill": rng.choice(["black", "#111", "#222"]),
        "background": rng.choice(["white", "#f7f7f7"]),
        "scale": rng.uniform(0.85, 0.95),
        "tx": rng.uniform(-2.0, 2.0),   # px
        "ty": rng.uniform(-2.0, 2.0),   # px
    }


# ===============================
# 🔥 Pillow 渲染单字 PNG（无系统依赖）
# ===============================

def render_hanzi_png(
    char: str,
    out_path: Path,
    size: int = 60,
    style: Dict[str, Any] | None = None,
    font_file: Path | None = None,
):
    if style is None:
        style = {}

    # ---- style ----
    base_font_size = style.get("font_size", 48)
    scale = style.get("scale", 1.0)
    font_size = int(base_font_size * scale)

    fill = style.get("fill", "black")
    background = style.get("background", "white")
    tx = style.get("tx", 0.0)
    ty = style.get("ty", 0.0)

    # ---- canvas ----
    img = Image.new("RGB", (size, size), color=background)
    draw = ImageDraw.Draw(img)

    # ---- font ----
    try:
        if font_size not in FONT_CACHE:
            FONT_CACHE[font_size] = ImageFont.truetype(
                font=str(font_file),
                size=font_size,
            )
        font = FONT_CACHE[font_size]
    except Exception as e:
        print(f"⚠️ 字体加载失败，使用默认字体: {e}", file=sys.stderr)
        font = ImageFont.load_default()

    # ---- bbox & center ----
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (size - text_w) / 2 + tx
    y = (size - text_h) / 2 + ty

    # ---- draw ----
    draw.text(
        (x, y),
        char,
        font=font,
        fill=fill,
        anchor="lt",
    )

    # ---- save ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

import shutil

# ===============================
# Main pipeline（回到你原始版本）
# ===============================

def generate_from_pairs_json(
    json_path: str,
    output_root: str,
    size: int = 60,
):
    json_path = Path(json_path)

    output_root = Path(output_root)

    # 如果已存在，先删除整个目录
    if output_root.exists():
        shutil.rmtree(output_root)

    # 重新创建
    output_root.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        pairs = json.load(f)

    assert isinstance(pairs, list), "JSON 顶层必须是 list"

    for pair_id, pair in enumerate(pairs):
        assert (
            isinstance(pair, list) and len(pair) >= 2
        ), f"pair {pair_id} 不是长度 ≥2 的列表: {pair}"

        # ⭐ pair-wise 可复现 style
        style = sample_pair_style(seed=pair_id)

        pair_dir = output_root / str(pair_id)
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        FONT_FILE = random.choice([
            p for p in FONT_DIR.iterdir()
            if p.suffix.lower() in {".ttf", ".ttc", ".otf"}
        ])
        
        for char in pair:
            if not isinstance(char, str) or len(char) != 1:
                print(f"⚠️ 跳过非法字符: {char} (pair {pair_id})")
                continue

            out_png = pair_dir / f"{char}.png"
            render_hanzi_png(
                char=char,
                out_path=out_png,
                size=size,
                style=style,
                font_file = FONT_FILE,
            )


# ===============================
# CLI
# ===============================

if __name__ == "__main__":
    generate_from_pairs_json(
        json_path="similar_chars.json",
        output_root="hanzi_png",
        size=60,
    )
