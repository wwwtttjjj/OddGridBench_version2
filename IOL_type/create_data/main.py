import numpy as np
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import random

from utils import (
    apply_odd_variations,
    lab_to_rgb,
    generate_lab_color,
    random_background_color,
    save_pair,
    ensure_dirs,
    rotate_block_keep_full,
    generate_local_odd_strength,
    resize_block_to_blocksize,
    move_position,
    add_gaussian_noise,
    _select_odd_positions,
    add_blur,
    save_visualized_odds,
)
from utils import *
from configs import configs, configs_odd, randomize_config
from shapes import draw_random_shape, draw_shape_by_name, register_all_svg


# ================================================================
# 全局配置：支持的 odd 类型 & 每个 odd 的强度采样范围
# ================================================================

# 每个 odd 块可从这里的类型中随机选 1~3 个组合
ALL_TYPES = ["color", "size", "rotation", "position", "blur", "occlusion","fracture","overlap"]
# ALL_TYPES = ["overlap"]

# ================================================================
# 基础构件：base block / odd 位置 / odd 类型 / odd 参数
# ================================================================

def _generate_base_block(block_size, background_rgb):
    """
    生成一张基础 block 的颜色 & 形状：
      - 随机 LAB 颜色
      - 转为 RGB
      - 在给定背景色上用 draw_random_shape 画出 base_shape

    返回:
        base_lab: (3,) LAB 颜色
        base_rgb: (3,) RGB 颜色 [0,1]
        base_shape: 形状名称字符串
    """
    base_lab = generate_lab_color()
    base_rgb = lab_to_rgb(base_lab)
    _, base_shape = draw_random_shape(
        block_size,
        color=base_rgb,
        bgcolor=background_rgb,
    )
    return base_lab, base_rgb, base_shape


def _generate_odd_types(n, max_attributes):
    """
    为每一个 odd 块生成一个类型列表（1~len(ALL_TYPES) 个类型，且不重复）。

    参数:
        n: odd 块数量

    返回:
        odd_types_per_block: 长度 n 的 list，每个元素是一个 list[str]，
                             例如 ["color"]、["size", "rotation"] 等。
    """
    odd_types_per_block = []
    for _ in range(n):
        # 每个 odd 随机选 k 个类型（k ∈ [1, len(ALL_TYPES)]）
        k = np.random.randint(1, max_attributes)
        types = np.random.choice(ALL_TYPES, size=k, replace=False).tolist()
        odd_types_per_block.append(types)
    return odd_types_per_block

def _generate_odd_parameters(
    base_shape,
    base_lab,
    base_rgb,
    base_angle,
    block_size,
    odd_types_per_block,
    configs_odd,
    args,
):
    """
    为每个 odd 块生成它自身的参数（颜色 / 大小 / 角度）。

    参数:
        base_shape:         基础形状名称
        base_lab:           基础 LAB 颜色
        base_rgb:           基础 RGB 颜色
        base_angle:         基础角度（来自 args.base_angle）
        block_size:         基础块大小
        odd_types_per_block:每个 odd 的类型列表（如 ["color", "rotation"]）
        configs_odd:        每个 odd 的强度采样配置
        args:               其他全局配置（用于 apply_odd_variations）

    返回:
        odd_params:        list[dict]，每个 odd 的具体属性
        final_base_angle:  用于整张图片中 base block 旋转的角度
                           （目前取第一个 odd 的 base_angle）
    """
    odd_params = []

    for odd_type_list in odd_types_per_block:
        # 每个 odd 自己的变化幅度（非常重要）
        local_de, local_size_ratio, local_angle_scale, local_position, local_blur, local_occlusion, local_fracture, local_overlap = generate_local_odd_strength(configs_odd)
        odd_lab, odd_rgb, odd_block_size, _, odd_base_angle, odd_angle, odd_position, odd_blur, odd_occlusion, odd_fracture, odd_overlap = apply_odd_variations(
            base_shape,
            base_lab,
            base_rgb,
            base_angle,
            block_size,
            odd_type_list,
            local_de,
            local_size_ratio,
            local_angle_scale,
            local_position,
            local_blur,
            local_occlusion,
            local_fracture,
            local_overlap,
            args,
        )
        # if "rotation" in odd_type_list:
        #     print(local_angle_scale, base_angle, odd_angle)
        # if "position" in odd_type_list:
        # print(odd_position)
        # print(odd_fracture)
        odd_params.append({
            "types": odd_type_list,
            "lab": odd_lab,
            "rgb": odd_rgb,
            "block_size": odd_block_size,
            "base_angle": odd_base_angle,
            "odd_angle": odd_angle,
            "delta_e": float(local_de),
            "size_ratio": float(local_size_ratio),
            "angle_strength": float(local_angle_scale),
            "odd_position":odd_position,
            "blur_scale": float(odd_blur),
            "occlusion_scale": float(odd_occlusion),
            "fracture_scale": float(odd_fracture),
            "overlap_scale": float(odd_overlap),
        })

    return odd_params, base_angle


# ================================================================
# 画布 & 单元格绘制
# ================================================================

def _create_canvas(grid_size, block_size, gap, margin, background_rgb):
    """
    根据 grid + block_size + gap + margin 计算整张图像大小，并创建背景图。

    参数:
        grid_size:     (h, w)
        block_size:    块大小
        gap:           块间距
        margin:        外边距
        background_rgb:背景颜色 (r,g,b) in [0,1]

    返回:
        img:   (H, W, 3) float32 图像
        img_h: 高度
        img_w: 宽度
    """
    h, w = grid_size

    core_h = h * block_size + (h - 1) * gap
    core_w = w * block_size + (w - 1) * gap

    img_h = core_h + 2 * margin
    img_w = core_w + 2 * margin

    img = np.ones((img_h, img_w, 3), dtype=np.float32) * np.array(background_rgb)

    return img, img_h, img_w


def _draw_cells(
    grid_size,
    block_size,
    gap,
    margin,
    odd_indices,
    odd_params,
    base_shape,
    base_rgb,
    background_rgb,
    base_angle,
    odd_types_per_block,
):
    """
    在画布上绘制所有格子：
      - 普通格子画 base block
      - odd 位置按 odd_params 中的信息画变化后的 block

    参数:
        grid_size:          (h, w)
        block_size:         基础块大小
        gap:                间距像素
        margin:             外边距
        odd_indices:        哪些 index 是 odd（list[int]）
        odd_params:         每个 odd 的具体参数（颜色、大小、角度等）
        base_shape:         基础形状
        base_rgb:           基础 RGB 颜色
        background_rgb:     背景 RGB
        base_angle:         base block 全局旋转角度
        odd_types_per_block:每个 odd 的类型列表（与 odd_params 对齐）

    返回:
        img:      画完所有格子的图
        odd_list: 用于写入 meta 的 odd 信息列表
    """
    h, w = grid_size
    total_cells = h * w

    # 创建画布
    img, img_h, img_w = _create_canvas(grid_size, block_size, gap, margin, background_rgb)

    odd_list = []

    # 是否存在 rotation 类型（只要有一个 odd 用到了 "rotation"）
    image_has_rotation = any("rotation" in t for t in odd_types_per_block)

    for idx in range(total_cells):
        i, j = divmod(idx, w)

        # 每个格子的左上角坐标
        cx = margin + j * (block_size + gap)
        cy = margin + i * (block_size + gap)

        # -------------------------------------------------
        # odd 格子
        # -------------------------------------------------
        if idx in odd_indices:
            k = odd_indices.index(idx)          # 取出当前 odd 的参数索引
            params = odd_params[k]
            odd_type_list = params["types"]

            bs = params["block_size"]
            color = params["rgb"]

            # 用变换后的尺寸 & 颜色画出 shape
            block_img, _ = draw_shape_by_name(
                base_shape,
                bs,
                color=color,
                bgcolor=background_rgb,
            )
            
            # 调整回统一的 block_size（通过裁剪 / 补边）
            block_img = resize_block_to_blocksize(block_img, block_size, background_rgb)
            # 调整position造成的位移
            block_img = move_position(block_img, block_size, background_rgb, params["odd_position"])
            # blur的逻辑
            block_img = add_blur(block_img, params["blur_scale"])
            # occlusion的逻辑
            block_img = add_occlusion(block_img, params["occlusion_scale"])
            # fracture的逻辑
            block_img = add_fracture(block_img, params["fracture_scale"], background_rgb)

            # overlap的逻辑
            block_img = add_overlap(block_img, params["overlap_scale"], background_rgb)
                                    
            # ------ 旋转逻辑 ------
            if image_has_rotation:
                if "rotation" in odd_type_list:
                    # rotation odd → 使用 odd_angle
                    block_img = rotate_block_keep_full(block_img, params["odd_angle"], background_rgb)
                else:
                    # 其它 odd（仅 color/size）→ 使用 base_angle
                    block_img = rotate_block_keep_full(block_img, params["base_angle"], background_rgb)

            # 记录 meta 信息
            odd_list.append({
                "types": odd_type_list,
                "row": i + 1,
                "col": j + 1,
                "bbox": {
                    "x": int(cx),
                    "y": int(cy),
                    "w": block_img.shape[1],
                    "h": block_img.shape[0],
                },
                "delta_e": params["delta_e"] if "color" in odd_type_list else None,
                "size_ratio": params["size_ratio"] if "size" in odd_type_list else None,
                "angle_strength": params["angle_strength"] if "rotation" in odd_type_list else None,
                "position_scale": params["odd_position"] if "position" in odd_type_list else None,
                "blur_scale": params["blur_scale"] if "blur" in odd_type_list else None,
                "occlusion_scale": params["occlusion_scale"] if "occlusion" in odd_type_list else None,
                "fracture_scale": params["fracture_scale"] if "fracture" in odd_type_list else None,
                "overlap_scale": params["overlap_scale"] if "overlap" in odd_type_list else None,
            })

        # -------------------------------------------------
        # base 格子
        # -------------------------------------------------
        else:
            bs = block_size
            color = base_rgb

            block_img, _ = draw_shape_by_name(
                base_shape,
                bs,
                color=color,
                bgcolor=background_rgb,
            )
            if image_has_rotation:
                block_img = rotate_block_keep_full(block_img, base_angle, background_rgb)

        # 将 block 贴到大图上
        img[cy:cy + block_img.shape[0], cx:cx + block_img.shape[1]] = block_img

        # debug：给所有 block 画黑框（随时注释）
        cv2.rectangle(
            img,
            (cx, cy),
            (cx + block_img.shape[1], cy + block_img.shape[0]),
            (0, 0, 0),
            1,
        )
    img = add_gaussian_noise(img, sigma=0.01)

        
    return img, odd_list


# ================================================================
# 元数据组装
# ================================================================

def _generate_metadata(
    grid_size,
    base_shape,
    base_lab,
    odd_count,
    odd_list,
    img_h,
    img_w,
    base_angle,
):
    """
    将关键信息打包成 meta 字典（写入 JSON）。
    """
    return {
        "grid_size": [grid_size[0], grid_size[1]],
        "base_shape": base_shape,
        "base_lab": base_lab.tolist(),
        "odd_count": odd_count,
        "odd_list": odd_list,
        "image_size": [img_h, img_w],
        "base_angle": base_angle,
    }


# ================================================================
# 核心入口：生成一张 odd-one-out 图像
# ================================================================

def generate_odd_one_out_image(
    grid_size,
    block_size,
    gap,
    margin,
    background_rgb,
    args,
):
    """
    生成一张 odd-one-out 图片：
      - 每个 odd 的类型为列表（支持 1~3 种组合）
      - 每个 odd 都有自己独立的变化强度（ΔE, size_ratio, angle_scale）
      - 旋转逻辑：只要有 rotation 类型的 odd，整图 base block 也按 base_angle 旋转
    """

    h, w = grid_size

    # gap 做一次安全取整
    gap = max(0, int(round(gap)))

    # 1) 生成基础块 & 基础颜色/形状
    base_lab, base_rgb, base_shape = _generate_base_block(block_size, background_rgb)


    odd_nums = [0, 1, 2, 3, 4]
    odd_pro = [0.1, 0.5, 0.2, 0.1,0.1]
    
    # 使用 random.choices 进行加权抽样
    # k=1 表示抽一个，返回的是列表，所以用 [0] 取出值
    chosen_odd_count = random.choices(odd_nums, weights=odd_pro, k=1)[0]
    total_cells, n, odd_indices = _select_odd_positions(grid_size, chosen_odd_count)

    # 3) 为每一个 odd 生成它的类型组合
    odd_types_per_block = _generate_odd_types(n, args.max_attributes + 1)

    # 4) 为每一个 odd 生成它自己的参数（颜色、大小、角度变化强度）
    odd_params, base_angle = _generate_odd_parameters(
        base_shape=base_shape,
        base_lab=base_lab,
        base_rgb=base_rgb,
        base_angle=args.base_angle,
        block_size=block_size,
        odd_types_per_block=odd_types_per_block,
        configs_odd=configs_odd,
        args=args,
    )

    # 5) 绘制整张图
    img, odd_list = _draw_cells(
        grid_size=grid_size,
        block_size=block_size,
        gap=gap,
        margin=margin,
        odd_indices=odd_indices,
        odd_params=odd_params,
        base_shape=base_shape,
        base_rgb=base_rgb,
        background_rgb=background_rgb,
        base_angle=base_angle,
        odd_types_per_block=odd_types_per_block,
    )

    # 6) 再创建一次画布只是为了拿 img_h / img_w（保持你原逻辑不变）
    _, img_h, img_w = _create_canvas(grid_size, block_size, gap, margin, background_rgb)

    # 7) 打包 meta
    meta = _generate_metadata(
        grid_size=grid_size,
        base_shape=base_shape,
        base_lab=base_lab,
        odd_count=n,
        odd_list=odd_list,
        img_h=img_h,
        img_w=img_w,
        base_angle=base_angle,
    )
    if args.rowcol_image:
        img_with_number = img.copy()  # 副本用于加行列编号
        img_with_number = add_row_col_numbers(img_with_number, (h, w), block_size, gap, margin, background_rgb)
    else:
        img_with_number = None
    # 当前不生成带行列编号的图，因此第二个返回值为 None（保持接口兼容）
    return img, img_with_number, meta


# ================================================================
# 单样本生成（并写入磁盘）
# ================================================================

def generate_single(idx, args, img_dir, meta_dir):
    """
    生成单张图像并保存：
      1) 复制 args 并随机化 configs
      2) 调用 generate_odd_one_out_image 得到 (img, meta)
      3) 调用 save_pair 写入 PNG + JSON
    """
    args_copy = copy.deepcopy(args)

    # 随机化 configs（grid, margin, block_size, gap, angle_sacle 等）
    cfg = randomize_config(configs)
    for k, v in cfg.items():
        setattr(args_copy, k, v)

    # 一些缺省字段的兜底初始化
    defaults = {
        "base_angle": 0,
        "rotation_banned": [],
        "angle_sacle": 0,
        "size_ratio": 1.0,
        "dx": 0.0,
        "dy": 0.0,
    }
    for k, v in defaults.items():
        if not hasattr(args_copy, k):
            setattr(args_copy, k, v)

    try:
        img, img_with_number, meta = generate_odd_one_out_image(
            grid_size=(args_copy.grid_y, args_copy.grid_x),
            block_size=args_copy.block_size,
            gap=args_copy.gap,
            margin=args_copy.margin,
            background_rgb=random_background_color(),
            args=args_copy,
        )

        meta["index"] = idx

        save_pair(
            image=img,
            meta=meta,
            img_dir=img_dir,
            meta_dir=meta_dir,
            index=idx,
            img_with_number=img_with_number,
            draw_bbox=args_copy.draw_bbox,
        )

        # ------ 可视化 odd 图案（调试用，可随时注释掉） ------
        # vis_path = os.path.join(img_dir, f"{idx:06d}_vis.png")
        # save_visualized_odds(img, meta["odd_list"], vis_path)

        return idx, True, None

    except Exception as e:
        return idx, False, str(e)


# ================================================================
# 数据集构建（并行，多进程）
# ================================================================

def build_dataset(args):
    """
    根据命令行参数并行生成整个数据集。
    """
    img_dir, meta_dir = ensure_dirs(args.data_type)
    num_workers = max(1, args.num_workers)

    print(f"🚀 Starting generation with {num_workers} workers, total {args.number} samples...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_single, idx, args, img_dir, meta_dir)
            for idx in range(args.number)
        ]

        for future in as_completed(futures):
            idx, success, msg = future.result()
            if success:
                print(f"[OK] Generated sample {idx}")
            else:
                print(f"[Warning] Sample {idx} failed: {msg}")

    print(f"✅ Finished generating {args.number} images into folder: {args.data_type}")


# ================================================================
# 命令行入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--data_type", type=str, default="test_data")
    parser.add_argument("--num_workers", type=int, default=16)

    # how many odds in each image (max)
    parser.add_argument("--max_num_odds", type=int, default=3)
    # how many attributes for each odd (max)
    parser.add_argument("--max_attributes", type=int, default=len(ALL_TYPES))
    

    args = parser.parse_args()

    args.draw_bbox = (args.data_type == "test_data")
    args.rowcol_image = (args.data_type == "test_data")
    
    if args.data_type == "val_data":
        register_all_svg(f"../../IOL_type/create_data/svg_file_test")
        
    # 注册SVG文件
    register_all_svg(f"../../IOL_type/create_data/svg_file_{args.data_type[:-5]}")
    
    build_dataset(args)
