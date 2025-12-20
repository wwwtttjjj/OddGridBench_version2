import random
import math
configs = {
    "icons_per_group" : [10, 20],
    "block_size":[100, 200],
    "base_angle":[0,360],
    "rotation_banned": ["symbolic(&)circle,natural(&)snowflake"]
}

# 每个 odd 自己的“变化强度”范围（与 args 无关，纯局部配置）
configs_odd = {
    "de_range": [5, 15],          # 每个 odd 的 ΔE 范围
    "size_range": [0.85, 1.15],   # 每个 odd 的 size_ratio 范围
    "size_offset": [0.95, 1.05],  # ❗ 排除区间（例如避免太接近 1.0）
    "angle_range": [-20, 20],     # 每个 odd 的 angle_scale 范围
    "angle_offset": [-5, 5],      # ❗ 排除区间（例如避免太小角度变化）
    "position_range":[0.85, 1.15],
    "position_offset":[0.95, 1.05],
    "blur_range":[0.6, 1.1],
    "occlusion_range":[0.05, 0.15],
    "fracture_range":[0.05, 0.2],
    "overlap_range":[0.5, 1],
}


def randomize_config(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            # 整数范围
            if all(isinstance(x, int) for x in v):
                out[k] = random.randint(v[0], v[1])
            # 浮点范围
            elif all(isinstance(x, (int, float)) for x in v):
                out[k] = round(random.uniform(v[0], v[1]), 2)
            else:
                out[k] = v
        else:
            out[k] = v
    
    # block_size = out["block_size"]
    # # 随机方向（左/右、上/下）
    # out["dx"] = math.ceil(block_size * out["dx"]) * random.choice([-1, 1])
    # out["dy"] = math.ceil(block_size * out["dy"]) * random.choice([-1, 1])
    
    # # --- angle_scale: 强制排除 [-min_offset, min_offset] 区间 ---
    # if "angle_sacle" in cfg and "angle_min_offset" in cfg:  # 注意 key 是 angle_sacle
    #     lo, hi = cfg["angle_sacle"]
    #     min_angle = cfg["angle_min_offset"]

    #     if random.random() < 0.5:
    #         out["angle_sacle"] = int(round(random.uniform(lo, -min_angle), 2))
    #     else:
    #         out["angle_sacle"] = int(round(random.uniform(min_angle, hi), 2))

    # # --- 随机抽取 odd_type ---
    # if "odd_type" in cfg:
    #     n = random.randint(1, len(cfg["odd_type"]))  # 随机选择 1~len 个
    #     out["odd_type"] = random.sample(cfg["odd_type"], n)

    # # --- size_ratio: 避开中间区域 (例如 0.95~1.05) ---
    # if "size_ratio" in cfg and "size_ratio_min_offset" in cfg:
    #     lo, hi = cfg["size_ratio"]
    #     min_offset = cfg["size_ratio_min_offset"]

    #     if random.random() < 0.5:
    #         out["size_ratio"] = round(random.uniform(lo, 1 - min_offset), 2)
    #     else:
    #         out["size_ratio"] = round(random.uniform(1 + min_offset, hi), 2)
    return out


