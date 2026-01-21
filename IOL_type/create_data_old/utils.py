import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import os
import json
import numpy
import shutil
import random
import math
import numpy as np
from PIL import Image, ImageDraw
import itertools
import cv2

def add_gaussian_noise(img, sigma=0.02):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0.0, 1.0)

def _select_odd_positions(grid_size, num_odds):
    """
    在 grid 上随机挑选 num_odds 个位置作为 odd。

    参数:
        grid_size: (h, w)
        num_odds:  目标 odd 数量

    返回:
        total_cells: 网格总 cell 数
        n:           实际采样到的 odd 数（夹在 [1, total_cells]）
        odd_indices: 长度为 n 的 list[int]，每个是一维 index（0 ~ total_cells-1）
    """
    h, w = grid_size
    total_cells = h * w

    # 至少 1 个，最多不超过格子总数
    n = max(1, min(num_odds, total_cells))
    odd_indices = np.random.choice(total_cells, size=n, replace=False).tolist()

    return total_cells, n, odd_indices


def generate_lab_color(l_range=(20, 70), a_range=(-40, 40), b_range=(-40, 40)):
    """随机生成一个更深的 LAB 颜色（避免白色或过亮），保留两位小数"""
    L = np.random.uniform(*l_range)
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    return np.round(np.array([L, a, b]), 2)


def perturb_color(base_lab, target_delta_e, step=1.0, max_iter=5000, tol=0.5):
    """
    生成与 base_lab 相差约 target_delta_e 的颜色 (ΔE2000)，保留两位小数
    """
    best_candidate = base_lab
    best_diff = 1e9
    for _ in range(max_iter):
        candidate = base_lab + np.random.uniform(-step, step, 3) * target_delta_e
        dE = color.deltaE_ciede2000(
            base_lab[np.newaxis, :], candidate[np.newaxis, :]
        )[0]
        diff = abs(dE - target_delta_e)
        if diff < best_diff:
            best_diff = diff
            best_candidate = candidate
        if diff < tol:
            break
    return np.round(best_candidate, 2)

def lab_to_rgb(lab):
    """LAB 转 RGB，并裁剪到 [0,1]"""
    rgb = color.lab2rgb(lab[np.newaxis, np.newaxis, :])
    return np.clip(rgb[0, 0, :], 0, 1)

def ensure_dirs(data_type: str):
    """
    创建 难度/image 与 难度/metadata 目录
    - 若已存在，则先清空再重新创建
    """
    img_dir = os.path.join(data_type, "image")
    meta_dir = os.path.join(data_type, "metadata")
    img_red_dir = os.path.join(data_type, "image_red")
    image_with_number_dir = os.path.join(data_type, "image_number")
    


    # 如果存在旧目录则先删除
    for d in [img_dir, meta_dir, img_red_dir, image_with_number_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print(f"Creating directories '{img_dir}', {img_red_dir}, and '{meta_dir}'...")

    # 重新创建空目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(img_red_dir, exist_ok=True)
    os.makedirs(image_with_number_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    

    return img_dir, meta_dir

def save_image_as_png(image: np.ndarray, path: str):
    """保存为 PNG 文件"""
    plt.imsave(path, image)


# def save_pair(image, meta, img_dir, meta_dir, index):
#     img_name = f"image_{index}.png"
#     meta_name = f"metadata_{index}.json"

#     img_path = os.path.join(img_dir, img_name)
#     meta_path = os.path.join(meta_dir, meta_name)

#     meta = dict(meta)  # 复制一份
#     meta["image_file"] = os.path.join("image", img_name)
#     meta["metadata_file"] = os.path.join("metadata", meta_name)
#     save_image_as_png(image, img_path)

#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)
def save_pair(image, meta, img_dir, meta_dir, index, img_with_number, draw_bbox=False):
    img_name = f"image_{index}.png"
    meta_name = f"metadata_{index}.json"

    img_path = os.path.join(img_dir, img_name)
    meta_path = os.path.join(meta_dir, meta_name)

    meta = dict(meta)  # 复制一份
    meta["image_file"] = os.path.join("image", img_name)
    meta["metadata_file"] = os.path.join("metadata", meta_name)
    save_image_as_png(image, img_path)

    # # ✅ 在这里画框
    # if draw_bbox and "odd_bbox" in meta:
    #     img_pil = Image.fromarray((image * 255).astype(np.uint8))
    #     draw = ImageDraw.Draw(img_pil)
    #     x, y, w, h = meta["odd_bbox"]["x"], meta["odd_bbox"]["y"], meta["odd_bbox"]["w"], meta["odd_bbox"]["h"]
    #     draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    #     image = np.asarray(img_pil, dtype=np.float32) / 255.0
    #     image_red_dir = img_dir.replace("image", "image_red")
    #     os.makedirs(image_red_dir, exist_ok=True)
    #     save_image_as_png(image, os.path.join(image_red_dir, img_name))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # img_with_number = None
    if img_with_number is not None:
        image_with_number_dir = img_dir.replace("image", "image_number")
        os.makedirs(image_with_number_dir, exist_ok=True)
        save_image_as_png(img_with_number, os.path.join(image_with_number_dir, img_name))

def apply_odd_variations(
    base_shape,
    base_lab,
    base_rgb,
    base_angle,
    block_size,
    odd_type_list,
    delta_e,        # ★ 每个 odd 自己的变化强度
    size_ratio,
    angle_scale,
    position_ration,
    local_blur,
    local_occlusion,
    local_fracture,
    local_overlap,
    args
):
    """
    返回：
    odd_lab, odd_rgb, odd_block_size, block_size, base_angle, odd_angle
    """

    # 确保 odd_type_list 是 list
    if isinstance(odd_type_list, str):
        odd_type_list = [odd_type_list]

    odd_lab = np.array(base_lab, dtype=np.float32).copy()
    odd_rgb = np.array(base_rgb, dtype=np.float32).copy()
    odd_block_size = int(block_size)
    odd_angle = float(base_angle)

    # ---- COLOR ----
    if "color" in odd_type_list:
        lab_delta = np.random.uniform(-delta_e, delta_e, size=3)
        odd_lab = odd_lab + lab_delta
        odd_rgb = lab_to_rgb(odd_lab)

    # ---- SIZE ----
    if "size" in odd_type_list:
        odd_block_size = max(1, int(round(block_size * size_ratio)))

    # ---- ROTATION ----
    if "rotation" in odd_type_list:
        odd_angle = float(base_angle + angle_scale) % 360

    # -----POSITION ---
    odd_position = [0, 0]
    if 'position' in odd_type_list:
        odd_position_x = int(position_ration[0] * block_size)
        odd_position_y = int(position_ration[1] * block_size)
        odd_position = [odd_position_x, odd_position_y]

    # -----BLUR ----
    blur_scale = 0.0
    if 'blur' in odd_type_list:
        blur_scale = float(local_blur)
        
    # ----- OCCLUSION ----
    occlusion_scale = 0.0
    if 'occlusion' in odd_type_list:
        occlusion_scale = float(local_occlusion)
    # ----- FRACTURE ----
    fracture_scale = 0.0
    if 'fracture' in odd_type_list:
        fracture_scale = float(local_fracture)
        
    # ----- overlap ----
    overlap_scale = 0.0
    if 'overlap' in odd_type_list:
        overlap_scale = float(local_overlap)
        
    return odd_lab, odd_rgb, odd_block_size, block_size, base_angle, odd_angle, odd_position, blur_scale, occlusion_scale, fracture_scale, overlap_scale

def get_block_position(i, j, block_size, gap, margin, img_w, img_h,
                       idx, odd_pos, odd_block_size, odd_type, args):
    """计算 block 的放置位置"""
    y0 = margin + i * (block_size + gap)
    x0 = margin + j * (block_size + gap)

    # odd_type: position 偏移
    if idx == odd_pos and "position" in odd_type:
        x0 = x0 + args.dx
        y0 = y0 + args.dy
 

    return int(x0), int(y0)


def rotate_block_keep_full(img_np, angle, bgcolor=(1,1,1)):
    """
    旋转图形：
    1) 扩大画布避免裁剪
    2) 在大画布中心旋转
    3) 再中心裁剪回原大小

    保证：
    - 不偏移
    - 不裁掉图形
    - 最终 block 大小不变
    """

    h, w = img_np.shape[:2]
    original_size = h  # == w

    # ---------- Step 1: 创建更大的正方形画布 ----------
    padded_size = int(math.ceil(original_size * math.sqrt(2)))

    bg_uint8 = [int(c*255) for c in bgcolor]

    big = np.full((padded_size, padded_size, 3), bg_uint8, dtype=np.uint8)

    # 将原图放到大图中心
    offset = (padded_size - original_size) // 2
    big[offset:offset+h, offset:offset+w] = (img_np * 255).astype(np.uint8)

    # ---------- Step 2: 以大图中心旋转 ----------
    center = (padded_size // 2, padded_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        big,
        M,
        (padded_size, padded_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_uint8
    )

    # ---------- Step 3: 裁剪回 block_size ----------
    start = (padded_size - original_size) // 2
    end = start + original_size
    cropped = rotated[start:end, start:end]

    return cropped.astype(np.float32) / 255.0

def compute_min_gap_rotation(block_size, base_angle, odd_angle):
    def scale(angle):
        theta = math.radians(angle % 180)   # 旋转对称性，周期 180°
        return abs(math.cos(theta)) + abs(math.sin(theta))

    f1 = scale(base_angle)
    f2 = scale(odd_angle)

    max_scale = max(f1, f2)  # 找放大效果最大的
    min_gap = math.ceil(block_size * (max_scale - 1)) + 2
    return min_gap


def get_safe_gap(block_size, odd_type, gap, args):
    min_gap = 0  # 默认值
    size_gap = 0
    if "size" in odd_type:
        block_size = math.ceil(max(block_size, block_size * args.size_ratio))
        size_gap = math.ceil(max(block_size * args.size_ratio - block_size, 0))

    # --- 单独情况 ---
    if "rotation" in odd_type and "size" not in odd_type and "position" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle)

    elif "size" in odd_type and "rotation" not in odd_type and "position" not in odd_type:
        min_gap = size_gap

    elif "position" in odd_type and "rotation" not in odd_type and "size" not in odd_type:
        min_gap = max(abs(args.dx), abs(args.dy))

    # --- 两两组合 ---
    elif "rotation" in odd_type and "size" in odd_type and "position" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) + size_gap

    elif "rotation" in odd_type and "position" in odd_type and "size" not in odd_type:
        min_gap = compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) + max(abs(args.dx), abs(args.dy))

    elif "size" in odd_type and "position" in odd_type and "rotation" not in odd_type:
        min_gap = max(abs(args.dx), abs(args.dy)) + size_gap

    # --- 三个都要 ---
    elif "rotation" in odd_type and "size" in odd_type and "position" in odd_type:
        min_gap = (compute_min_gap_rotation(block_size, args.base_angle,args.base_angle + args.angle_sacle) +
                   size_gap +
                   max(abs(args.dx), abs(args.dy)))

    # --- gap 修正 ---
    if gap < min_gap:
        gap = min_gap
    return gap



def random_background_color(prob_white: float = 0.5,
                            light_range: tuple = (0.8, 1.0),
                            smooth: bool = True):
    """
    随机生成背景颜色（RGB）
    -------------------------------------
    Args:generate_odd_type_list
        prob_white: 保持白色背景的概率（默认 0.5）
        light_range: 淡色通道取值范围（默认 0.8~1.0）
        smooth: 若为 True，则三通道相差较小，颜色柔和

    Returns:
        tuple: (r, g, b)，范围在 [0, 1]
    """

    # 以 prob_white 概率返回白色
    if random.random() < prob_white:
        return (1.0, 1.0, 1.0)

    lo, hi = light_range

    if smooth:
        # 柔和淡色：三通道在 base±variation 范围内微调
        base = random.uniform(lo, hi)
        variation = 0.05
        r = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
        g = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
        b = round(min(1.0, max(0.0, base + random.uniform(-variation, variation))), 2)
    else:
        # 独立随机通道
        r = round(random.uniform(lo, hi), 2)
        g = round(random.uniform(lo, hi), 2)
        b = round(random.uniform(lo, hi), 2)

    return (r, g, b)



def generate_odd_type_list (base_types, total_number: int):
    odd_type_list = []

    # 平均分成7份
    n_per_group = total_number // 7
    remainder = total_number % 7  # 多退少补部分

    # 1️⃣ 单类别组合
    for t in base_types:
        odd_type_list.extend([[t]] * n_per_group)

    # 2️⃣ 随机两个类别组合
    two_combos = list(itertools.combinations(base_types, 2))
    for combo in random.choices(two_combos, k=n_per_group):
        odd_type_list.append(list(combo))

    # 3️⃣ 随机三个类别组合
    three_combos = list(itertools.combinations(base_types, 3))
    for combo in random.choices(three_combos, k=n_per_group):
        odd_type_list.append(list(combo))

    # 4️⃣ 四个类别组合
    four_combo = [base_types]
    odd_type_list.extend([four_combo[0]] * n_per_group)

    # 5️⃣ 补齐或裁剪到 total_number
    while len(odd_type_list) < total_number:
        odd_type_list.append(random.choice(odd_type_list))
    odd_type_list = odd_type_list[:total_number]

    random.shuffle(odd_type_list)
    return odd_type_list

def add_row_col_numbers(img, grid_size, block_size, gap, margin, background_rgb):
    h, w = grid_size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, block_size / 100)
    font_thickness = max(1, int(block_size / 30))
    text_color = (0, 0, 0) if np.mean(background_rgb) > 0.5 else (1, 1, 1)
    text_color = tuple(int(c * 255) for c in text_color)

    # --- 列编号（上方） ---
    for j in range(w):
        x0 = margin + j * (block_size + gap) + block_size // 2
        y0 = int(margin * 0.6)
        # 横向稍微往右偏移（+0.2），避免偏左
        x_shift = int(block_size * 0.1)
        # 不动纵向，因为是横行
        cv2.putText(
            img, str(j + 1),
            (x0 - int(block_size * 0.1) + x_shift, y0),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA
        )

    # --- 行编号（左侧） ---
    for i in range(h):
        x0 = int(margin * 0.3)
        y0 = margin + i * (block_size + gap) + block_size // 2
        # 纵向稍微往下偏移（+0.25），避免偏上
        y_shift = int(block_size * 0.2)
        cv2.putText(
            img, str(i + 1),
            (x0, y0 + y_shift),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA
        )

    return img

def sample_excluding_range(low, high, exclude_low, exclude_high):
    """
    从 [low, high] 内随机采样，但排除 [exclude_low, exclude_high] 的值。
    如果无 offset，则直接 uniform 采样。
    """
    # 没有 offset 的情况
    if exclude_low is None or exclude_high is None:
        return np.random.uniform(low, high)

    # 如果排除区间不在范围内，直接 uniform
    if exclude_low <= low or exclude_high >= high:
        return np.random.uniform(low, high)

    # Two-side sampling: 左半段 or 右半段选一段采样
    if random.random() < 0.5:
        # 左区间 [low, exclude_low]
        return np.random.uniform(low, exclude_low)
    else:
        # 右区间 [exclude_high, high]
        return np.random.uniform(exclude_high, high)


def generate_local_odd_strength(configs_odd):
    """
    返回单个 odd 的三个变化参数：
      - local_de
      - local_size_ratio
      - local_angle_scale
    支持 offset 排除区间。
    """

    # ΔE 没有 offset
    de_min, de_max = configs_odd["de_range"]
    local_de = int(np.random.uniform(de_min, de_max))

    # size_ratio（支持 offset）
    size_min, size_max = configs_odd["size_range"]
    size_offset = configs_odd.get("size_offset", None)
    if size_offset is not None:
        size_ex_lo, size_ex_hi = size_offset
    else:
        size_ex_lo = size_ex_hi = None

    local_size_ratio = sample_excluding_range(size_min, size_max, size_ex_lo, size_ex_hi)

    # angle_scale（支持 offset）
    angle_min, angle_max = configs_odd["angle_range"]
    angle_offset = configs_odd.get("angle_offset", None)
    if angle_offset is not None:
        angle_ex_lo, angle_ex_hi = angle_offset
    else:
        angle_ex_lo = angle_ex_hi = None

    local_angle_scale = sample_excluding_range(angle_min, angle_max, angle_ex_lo, angle_ex_hi)


    # position (offset）
    size_min, size_max = configs_odd["position_range"]
    size_offset = configs_odd.get("position_offset", None)
    if size_offset is not None:
        size_ex_lo, size_ex_hi = size_offset
    else:
        size_ex_lo = size_ex_hi = None

    position_x = sample_excluding_range(size_min, size_max, size_ex_lo, size_ex_hi)
    position_y = sample_excluding_range(size_min, size_max, size_ex_lo, size_ex_hi)

    # blur
    blur_min, blur_max = configs_odd["blur_range"]
    blur_scale = np.random.uniform(blur_min, blur_max)

    # occlusion
    occlusion_min, occlusion_max = configs_odd["occlusion_range"]
    occlusion_scale = np.random.uniform(occlusion_min, occlusion_max)
    
    # fracture
    fracture_min, fracture_max = configs_odd["fracture_range"]
    fracture_scale = np.random.uniform(fracture_min, fracture_max)
    
    # overlap
    overlap_min, overlap_max = configs_odd["overlap_range"]
    overlap_scale = np.random.uniform(overlap_min, overlap_max)
    
    return local_de, round(local_size_ratio,2), int(local_angle_scale), [round(position_x, 2) - 1, round(position_y, 2) - 1], round(blur_scale,2), round(occlusion_scale,2), round(fracture_scale,2), round(overlap_scale,2)

def resize_block_to_blocksize(block_img, target_size, background_rgb):
    """
    将 block_img 调整为 target_size × target_size：
    - 如果 block_img 太大：裁剪（中心裁剪）
    - 如果 block_img 太小：补边（使用背景色）
    """
    h, w, _ = block_img.shape
    if h == target_size and w == target_size:
        return block_img
    # -------- 情况1：block 变大，需要裁剪 --------
    if h > target_size or w > target_size:
        # 计算裁剪起点（中心裁剪）
        start_y = max(0, (h - target_size) // 2)
        start_x = max(0, (w - target_size) // 2)
        return block_img[start_y:start_y + target_size, start_x:start_x + target_size]

    # -------- 情况2：block 变小，需要补边 --------
    pad_y = target_size - h
    pad_x = target_size - w

    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    pad_left = pad_x // 2
    pad_right = pad_x - pad_left


    # 使用背景色创建填充
    bg = np.array(background_rgb, dtype=np.float32)

    padded = np.full((target_size, target_size, 3), bg, dtype=np.float32)

    padded[pad_top:pad_top + h, pad_left:pad_left + w] = block_img

    return padded


def save_visualized_odds(img, odd_list, save_path, color=(1.0, 0.0, 0.0), thickness=3):
    """
    给 odd 的 bbox 画红框并保存，用于可视化检查。
    img: float32 RGB, 0~1
    odd_list: meta["odd_list"]
    save_path: 输出路径
    """

    # 转 uint8
    vis = (img * 255).astype(np.uint8).copy()

    for odd in odd_list:
        x = odd["bbox"]["x"]
        y = odd["bbox"]["y"]
        w = odd["bbox"]["w"]
        h = odd["bbox"]["h"]

        # 如果你的图像是 RGB，cv2 需要 BGR
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

        cv2.rectangle(vis, (x, y), (x + w, y + h), bgr_color, thickness)

    cv2.imwrite(save_path, vis)

def move_position(block_img, block_size, background_rgb, odd_position):
    """
    根据 odd_position = [dx, dy] 对 block_img 做平移，
    并保持输出尺寸仍为 block_size × block_size。

    规则：
    - 正 dx：向右移动
    - 负 dx：向左移动
    - 正 dy：向下移动
    - 负 dy：向上移动
    - 超出部分丢弃，空白部分用 background_rgb 填充
    """

    dx, dy = odd_position  # 像素级偏移（可正可负）
    dx = int(round(dx))
    dy = int(round(dy))
    
    if dx == dy == 0:
        return block_img
    
    h, w, _ = block_img.shape
    assert h == block_size and w == block_size, "move_position 需要输入已经对齐到 block_size"

    # 创建背景画布
    bg = np.array(background_rgb, dtype=np.float32)
    shifted = np.full((block_size, block_size, 3), bg, dtype=np.float32)

    # 计算 source 与 target 的拷贝区域
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(block_size, block_size - dx)
    src_y1 = min(block_size, block_size - dy)

    tgt_x0 = max(0, dx)
    tgt_y0 = max(0, dy)
    tgt_x1 = tgt_x0 + (src_x1 - src_x0)
    tgt_y1 = tgt_y0 + (src_y1 - src_y0)

    # 执行拷贝
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[tgt_y0:tgt_y1, tgt_x0:tgt_x1] = block_img[src_y0:src_y1, src_x0:src_x1]

    return shifted

def add_blur(block_img, blur_scale):
    """
    使用 blur_scale ∈ [0,1] 直接控制 sigma（连续物理模糊强度）
    """
    if blur_scale <= 0:
        return block_img

    # ✅ 将 0~1 映射到合理的物理 sigma 范围
    # 一般工业级建议：sigma ∈ [0.5, 4.0]
    sigma = blur_scale

    # ✅ k 可以交给 OpenCV 自动推断（设为 0）
    blurred = cv2.GaussianBlur(block_img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return blurred

def add_occlusion(block_img, scale, color=None, cell_size=3):
    """
    随机在整张 block 上掉落若干个 n×n 小遮挡块：
    - scale ∈ [0, 1] 控制遮挡块数量（相对最大可放置数量）
    - cell_size: 小遮挡块尺寸（默认为 3 像素）
    - 其中一半优先落在图像中心区域
    """

    if scale <= 0:
        return block_img

    h, w, _ = block_img.shape
    cell_size = int(cell_size)

    if cell_size <= 0 or cell_size > min(h, w):
        return block_img

    # 最大可以不重叠放多少个 cell（粗略估计）
    max_cells = max(1, (h * w) // (cell_size * cell_size))

    # 实际要掉落多少个遮挡块（你原来的强度控制）
    num_drop = max(1, int(round(scale * max_cells)) // 4)

    # ===== 新增：一半中心，一半全局 =====
    num_center = num_drop // 2
    num_global = num_drop - num_center

    # 遮挡颜色：默认用整块均值
    if color is None:
        color = block_img.mean(axis=(0, 1))

    # ===== 1️⃣ 中心区域掉落 =====
    center_size = h // 2   # 中心正方形边长
    cy0 = (h - center_size) // 2
    cx0 = (w - center_size) // 2

    for _ in range(num_center):
        y0 = np.random.randint(cy0, cy0 + center_size - cell_size + 1)
        x0 = np.random.randint(cx0, cx0 + center_size - cell_size + 1)

        block_img[y0:y0 + cell_size, x0:x0 + cell_size] = color

    # ===== 2️⃣ 全图随机掉落 =====
    for _ in range(num_global):
        y0 = np.random.randint(0, h - cell_size + 1)
        x0 = np.random.randint(0, w - cell_size + 1)

        block_img[y0:y0 + cell_size, x0:x0 + cell_size] = color

    return block_img

def add_fracture(block_img, scale=0.3, bgcolor=(1, 1, 1),direction=None):
    """
    断裂异常（Fracture Anomaly）：
    - 在 block 中间制造“断裂 + 拉开”
    - 中间用背景色填充
    - 保持输出尺寸不变（H×W 不变）
    
    参数：
        block_img: float32, [H, W, 3], 取值 0~1
        scale: ∈ [0, 1]，控制裂开的强度（位移比例）
        direction: "vertical" / "horizontal" / None(随机)
        bgcolor: 背景色 (float RGB, 0~1)
    """

    if scale <= 0:
        return block_img

    H, W, _ = block_img.shape
    out = np.full_like(block_img, bgcolor, dtype=np.float32)

    # ---------- 1️⃣ 选择断裂方向 ----------
    if direction is None:
        direction = np.random.choice(["vertical", "horizontal"])

    # ---------- 2️⃣ 计算位移像素 ----------
    max_shift = int(min(H, W) * 0.25)   # 防止裂太狠
    shift = max(1, int(scale * max_shift))

    # ============================
    # ✅ 垂直断裂（左右分离）
    # ============================
    if direction == "vertical":
        cut = np.random.randint(W // 3, 2 * W // 3)

        left = block_img[:, :cut]      # 左半部分
        right = block_img[:, cut:]     # 右半部分

        # ---- 左侧向左拉 ----
        left_dst_x = max(0, -shift)
        left_src_x = max(0, shift)
        out[:, left_dst_x:left_dst_x + left.shape[1] - left_src_x] = left[:, left_src_x:]

        # ---- 右侧向右拉 ----
        right_dst_x = min(W - right.shape[1], cut + shift)
        out[:, right_dst_x:right_dst_x + right.shape[1]] = right

    # ============================
    # ✅ 水平断裂（上下分离）
    # ============================
    else:
        cut = np.random.randint(H // 3, 2 * H // 3)

        top = block_img[:cut, :]
        bottom = block_img[cut:, :]

        # ---- 上半部分向上拉 ----
        top_dst_y = max(0, -shift)
        top_src_y = max(0, shift)
        out[top_dst_y:top_dst_y + top.shape[0] - top_src_y, :] = top[top_src_y:, :]

        # ---- 下半部分向下拉 ----
        bottom_dst_y = min(H - bottom.shape[0], cut + shift)
        out[bottom_dst_y:bottom_dst_y + bottom.shape[0], :] = bottom

    return out

def add_overlap(
    block_img,
    scale,
    bgcolor,
    direction = None,
):
    """
    同形状错位重叠异常（Self-overlap / Ghosting Anomaly）：
    - 对当前 block 自身做一个小位移复制
    - 与原 block 发生 alpha 重叠
    - 形成“叠影 / 重影 / 错位重叠”效果
    - 输出尺寸保持不变

    参数：
        block_img: float32 [H, W, 3], 0~1
        scale: ∈ [0,1]，控制位移幅度（相对 block_size）
        alpha: ∈ (0,1)，控制重叠强度
        direction: "x" / "y" / None(随机)
        bgcolor: 背景色
    """
    alpha=random.uniform(0.6, 0.8)

    if scale <= 0:
        return block_img

    H, W, _ = block_img.shape

    # ---------- 1️⃣ 决定偏移方向 ----------
    if direction is None:
        direction = np.random.choice(["x", "y", "xy"])

    # ---------- 2️⃣ 计算像素级偏移 ----------
    max_shift = int(min(H, W) * 0.1)
    shift = max(1, int(scale * max_shift))

    if direction == "x":
        dx = np.random.choice([-shift, shift])
        dy = 0
    elif direction == "y":
        dx = 0
        dy = np.random.choice([-shift, shift])
    else:  # "xy"
        dx = np.random.choice([-shift, shift])
        dy = np.random.choice([-shift, shift])

    # ---------- 3️⃣ 构造平移后的副本 ----------
    shifted = np.full_like(block_img, bgcolor, dtype=np.float32)

    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)

    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = block_img[src_y0:src_y1, src_x0:src_x1]

    # ---------- 4️⃣ 执行 alpha 重叠融合 ----------
    out = (1 - alpha) * block_img + alpha * shifted

    return out