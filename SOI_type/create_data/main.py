import numpy as np
import argparse
import copy
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import random

# 导入你原有的工具函数
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
    add_blur,
)
from utils import *
from configs import configs, configs_odd, randomize_config
from shapes import draw_random_shape, draw_shape_by_name, register_all_svg

# 全局配置
ALL_TYPES = ["color", "size", "rotation", "position", "blur", "occlusion","fracture","overlap"]

# --------------------------- 保存单个icon（组内序号） ---------------------------
def save_group_icon(icon_img, icon_idx_in_group, group_img_dir):
    """
    保存组内单个icon
    :param icon_img: icon图像
    :param icon_idx_in_group: 组内序号（从1开始）
    :param group_img_dir: 组图片目录
    """
    # 转换为uint8并保存
    icon_img_uint8 = (np.clip(icon_img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(group_img_dir, f"{icon_idx_in_group}.png"), icon_img_uint8)

# --------------------------- 保存组元数据 ---------------------------
def save_group_metadata(group_info, save_metadir, group_idx):
    """
    保存整组的元数据（记录组内哪些是odd）
    :param group_info: 组信息字典
    :param save_metadir: 根目录
    :param group_idx: 组序号（从1开始）
    """
    # 保存组元数据
    with open(os.path.join(save_metadir, f"group_{group_idx}.json"), 'w', encoding='utf-8') as f:
        json.dump(group_info, f, indent=4)

# --------------------------- 基础生成函数 ---------------------------
def _generate_base_block(block_size, background_rgb):
    base_lab = generate_lab_color()
    base_rgb = lab_to_rgb(base_lab)
    _, base_shape = draw_random_shape(
        block_size,
        color=base_rgb,
        bgcolor=background_rgb,
    )
    return base_lab, base_rgb, base_shape

def _generate_odd_types(n, max_attributes):
    odd_types_per_block = []
    for _ in range(n):
        k = np.random.randint(1, max_attributes)
        types = np.random.choice(ALL_TYPES, size=k, replace=False).tolist()
        odd_types_per_block.append(types)
    return odd_types_per_block

def _generate_odd_parameters(
    base_shape, base_lab, base_rgb, base_angle, block_size, odd_types_per_block, configs_odd, args
):
    odd_params = []
    for odd_type_list in odd_types_per_block:
        local_de, local_size_ratio, local_angle_scale, local_position, local_blur, local_occlusion, local_fracture, local_overlap = generate_local_odd_strength(configs_odd)
        odd_lab, odd_rgb, odd_block_size, _, odd_base_angle, odd_angle, odd_position, odd_blur, odd_occlusion, odd_fracture, odd_overlap = apply_odd_variations(
            base_shape, base_lab, base_rgb, base_angle, block_size, odd_type_list,
            local_de, local_size_ratio, local_angle_scale, local_position, local_blur,
            local_occlusion, local_fracture, local_overlap, args
        )
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

# --------------------------- 选择odd位置（适配自定义数量） ---------------------------
def _select_odd_positions_custom(total_count, max_odds):
    """
    自定义选择odd位置（不再依赖grid）
    :param total_count: 总图标数
    :param max_odds: 最大odd数量
    :return: odd_indices: odd图标的索引列表（0-based）
    """
    # 随机选择1~max_odds个odd
    num_odds = random.randint(1, min(max_odds, total_count))
    # 随机选择位置
    odd_indices = random.sample(range(total_count), num_odds)
    return total_count, num_odds, odd_indices

# --------------------------- 生成单组图标 ---------------------------
def generate_single_group(group_idx, args, save_root, meta_dir):
    """
    生成单组图标（用icons_per_group控制数量）
    :param group_idx: 组序号（从1开始）
    :param args: 配置参数
    :param save_root: 保存根目录
    :param meta_dir: 元数据目录
    :return: 生成状态
    """
    try:
        # 1. 初始化配置
        args_copy = copy.deepcopy(args)
        cfg = randomize_config(configs)
        for k, v in cfg.items():
            # 只复制非grid相关的配置
            if not k.startswith('grid_'):
                setattr(args_copy, k, v)
        
        # 缺省参数兜底
        defaults = {
            "base_angle": 0, 
            "rotation_banned": [], 
            "angle_sacle": 0, 
            "size_ratio": 1.0, 
            "dx": 0.0, 
            "dy": 0.0,
            "block_size": cfg.get("block_size", 64)  # 默认block_size
        }
        for k, v in defaults.items():
            if not hasattr(args_copy, k):
                setattr(args_copy, k, v)
        
        # 2. 创建组目录（image1/image2...）
        group_name = f"image{group_idx}"
        group_img_dir = os.path.join(save_root, group_name)
        os.makedirs(group_img_dir, exist_ok=True)
        
        # 3. 基础配置
        block_size = args_copy.block_size
        background_rgb = random_background_color()
        total_icons = args_copy.icons_per_group  # 直接使用输入的图标数量
        
        # 4. 生成基础样式
        base_lab, base_rgb, base_shape = _generate_base_block(block_size, background_rgb)
        
        # 5. 选择odd位置和生成odd参数（自定义数量版本）
        # 每组的odd数量（1~max_num_odds）
        total_count, num_odds_in_group, odd_indices = _select_odd_positions_custom(
            total_icons, 
            args_copy.max_num_odds
        )
        
        # 生成odd类型和参数
        odd_types_per_block = _generate_odd_types(num_odds_in_group, args_copy.max_attributes + 1)
        odd_params, base_angle = _generate_odd_parameters(
            base_shape, base_lab, base_rgb, args_copy.base_angle, block_size, 
            odd_types_per_block, configs_odd, args_copy
        )
        
        # 构建odd映射（图标索引 -> odd参数）
        odd_index_map = {idx: params for idx, params in zip(odd_indices, odd_params)}
        image_has_rotation = any("rotation" in t for t in odd_types_per_block)
        
        # 6. 组元数据初始化
        group_info = {
            "group_name": group_name,
            "group_idx": group_idx,
            "total_icons": total_icons,
            "num_odds": num_odds_in_group,
            "block_size":block_size,
            "base_config": {
                "block_size": block_size,
                "base_shape": base_shape,
                "base_lab": base_lab.tolist(),
                "base_rgb": base_rgb.tolist(),
                "base_angle": base_angle
            },
            "odd_icons": [],  # 记录组内哪些是odd
            # "normal_icons": []  # 记录组内普通图标
        }
        
        # 7. 生成并保存组内每个图标（组内序号从1开始）
        for icon_idx in range(total_icons):
            # 组内序号（1,2,3...）
            icon_idx_in_group = icon_idx + 1
            
            # 生成单个图标
            if icon_idx in odd_indices:
                # Odd图标
                params = odd_index_map[icon_idx]
                odd_type_list = params["types"]
                bs = params["block_size"]
                color = params["rgb"]
                
                block_img, _ = draw_shape_by_name(base_shape, bs, color=color, bgcolor=background_rgb)
                block_img = resize_block_to_blocksize(block_img, block_size, background_rgb)
                block_img = move_position(block_img, block_size, background_rgb, params["odd_position"])
                block_img = add_blur(block_img, params["blur_scale"])
                block_img = add_occlusion(block_img, params["occlusion_scale"])
                block_img = add_fracture(block_img, params["fracture_scale"], background_rgb)
                block_img = add_overlap(block_img, params["overlap_scale"], background_rgb)

                if image_has_rotation:
                    block_img = rotate_block_keep_full(block_img, params["odd_angle"], background_rgb)
                
                # 添加高斯噪声
                block_img = add_gaussian_noise(block_img, sigma=0.01)
                
                # 记录odd图标信息
                odd_icon_info = {
                    "icon_name": f"{icon_idx_in_group}.png",
                    "icon_idx_in_group": icon_idx_in_group,
                    "icon_idx_0based": icon_idx,
                    "odd_types": odd_type_list,
                    "delta_e": params["delta_e"] if "color" in odd_type_list else None,
                    "size_ratio": params["size_ratio"] if "size" in odd_type_list else None,
                    "angle_strength": params["angle_strength"] if "rotation" in odd_type_list else None,
                    "position_scale": params["odd_position"] if "position" in odd_type_list else None,
                    "blur_scale": params["blur_scale"] if "blur" in odd_type_list else None,
                    "occlusion_scale": params["occlusion_scale"] if "occlusion" in odd_type_list else None,
                    "fracture_scale": params["fracture_scale"] if "fracture" in odd_type_list else None,
                    "overlap_scale": params["overlap_scale"] if "overlap" in odd_type_list else None,
                }
                group_info["odd_icons"].append(odd_icon_info)
                
            else:
                # 普通图标
                bs = block_size
                color = base_rgb
                block_img, _ = draw_shape_by_name(base_shape, bs, color=color, bgcolor=background_rgb)
                
                if image_has_rotation:
                    block_img = rotate_block_keep_full(block_img, base_angle, background_rgb)
                
                # 添加高斯噪声
                block_img = add_gaussian_noise(block_img, sigma=0.01)
                
                # # 记录普通图标信息
                # normal_icon_info = {
                #     "icon_name": f"{icon_idx_in_group}.png",
                #     "icon_idx_in_group": icon_idx_in_group,
                #     "icon_idx_0based": icon_idx
                # }
                # group_info["normal_icons"].append(normal_icon_info)
            
            # 保存组内图标（1.png, 2.png...）
            save_group_icon(block_img, icon_idx_in_group, group_img_dir)
        
        # 8. 保存组元数据
        save_group_metadata(group_info, meta_dir, group_idx)
        
        return group_idx, True, f"组 {group_name} 生成完成，共{total_icons}个图标（{num_odds_in_group}个odd）"
        
    except Exception as e:
        return group_idx, False, f"组 {group_idx} 生成失败: {str(e)}"

# --------------------------- 构建数据集 ---------------------------
def build_dataset(args):
    """
    构建数据集：
    - 每组一个目录（image1/image2...）
    - 组内图标：1.png, 2.png...（数量由icons_per_group控制）
    - 元数据：metadata/group_1.json, group_2.json...
    """
    # 根目录
    img_dir, meta_dir = ensure_dirs(args.data_type)
    num_workers = max(1, args.num_workers)
    total_groups = args.number  # 要生成的总组数
    

    
    # 并行生成各组
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for group_idx in range(1, total_groups + 1):  # 组序号从1开始
            futures.append(executor.submit(generate_single_group, group_idx, args, img_dir, meta_dir))
        
        # 收集结果
        success_count = 0
        fail_count = 0
        for future in as_completed(futures):
            group_idx, success, msg = future.result()
            if success:
                success_count += 1
                print(f"[OK] {msg}")
            else:
                fail_count += 1
                print(f"[ERROR] {msg}")

# --------------------------- 命令行参数 ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument("--number", type=int, default=10, help="要生成的总组数（image1~imageN）")
    parser.add_argument("--data_type", type=str, default="test_data", help="数据集名称（一级目录）")
    
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--max_num_odds", type=int, default=5, help="每组中最大odd图标数量")
    parser.add_argument("--max_attributes", type=int, default=1, help="每个odd图标的最大属性数")

    args = parser.parse_args()
    
    # 注册SVG文件
    register_all_svg(f"../../create_data/svg_file_{args.data_type[:-5]}")
    
    # 构建数据集
    build_dataset(args)