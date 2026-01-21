import os, io
import numpy as np
from PIL import Image
import cairosvg
from .registry import register_shape, shape_registry
import cv2

def add_gaussian_noise(img, sigma=0.02):
    """
    给 block 添加轻微高斯噪声
    img: float32, [0,1]
    sigma: 噪声强度，推荐 0.01 ~ 0.05
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0.0, 1.0)


def rasterize_svg(svg_str, block_size, color=(0,0,0), bgcolor=(1,1,1),
                  shrink_ratio=0.75):
    """
    渲染 SVG → numpy(H,W,3)
    shrink_ratio: 渲染后对图形再额外缩放的比例（例如 0.75 表示缩小到 75%）
    """

    # ---- Step 1: 正常渲染 SVG 到 block_size ----
    png_data = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=block_size,
        output_height=block_size
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 4

    alpha = arr[..., 3:4]
    fg_color = np.array(color)[None, None, :]
    bg_color = np.array(bgcolor)[None, None, :]

    rendered = fg_color * alpha + bg_color * (1 - alpha)   # block_size × block_size × 3

    # ---- Step 2: 缩小图形（留出 padding）----
    target = int(block_size * shrink_ratio)
    img_small = cv2.resize((rendered * 255).astype(np.uint8),
                           (target, target),
                           interpolation=cv2.INTER_AREA)

    img_small = img_small.astype(np.float32) / 255.0

    # ---- Step 3: 放到 block_size × block_size 画布中心 ----
    canvas = np.full((block_size, block_size, 3), bgcolor, dtype=np.float32)

    top = (block_size - target) // 2
    left = (block_size - target) // 2

    canvas[top:top + target, left:left + target] = img_small
    canvas = add_gaussian_noise(canvas)
    return canvas


def register_all_svg(folder):
    """递归扫描并注册 folder 下所有子目录中的 SVG 文件"""
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(".svg"):
                continue

            # 用子目录名 + 文件名作为注册名，避免重名
            rel_path = os.path.relpath(root, folder)  # 相对路径
            if rel_path == ".":
                shape_name = os.path.splitext(fname)[0]
            else:
                shape_name = f"{rel_path}(&){os.path.splitext(fname)[0]}"

            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                svg_str = f.read()

            # 注册函数
            def make_func(svg_str, shape_name):
                @register_shape(shape_name)
                def shape_func(block_size, color=(0,0,0), bgcolor=(1,1,1)):
                    return rasterize_svg(svg_str, block_size, color, bgcolor)
                return shape_func

            make_func(svg_str, shape_name)

    print(f"已注册 {len(shape_registry)} 个 SVG 图案: {list(shape_registry.keys())}")
    
# register_all_svg("/data/wengtengjin/colorsense/create_data/svg_file_test/")
# register_all_svg("/data/wengtengjin/colorsense/create_data/svg_file_train/")
