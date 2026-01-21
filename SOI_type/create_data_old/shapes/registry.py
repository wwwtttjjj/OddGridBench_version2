# shapes/registry.py
import random

shape_registry = {}

def register_shape(name):
    def decorator(func):
        shape_registry[name] = func
        return func
    return decorator

def draw_shape_by_name(name, block_size, color, bgcolor):
    func = shape_registry[name]
    return func(block_size, color=color, bgcolor=bgcolor), name

def draw_random_shape(block_size, color, bgcolor, allow=None, exclude=None):
    # 延迟导入 set_current_char 避免循环

    candidates = allow if allow else list(shape_registry.keys())

    # --- 新增：排除不允许的形状 ---
    if exclude:
        candidates = [c for c in candidates if c not in exclude]

    if not candidates:
        raise ValueError("No shapes available after applying allow/exclude filters!")

    name = random.choice(candidates)

    img = shape_registry[name](block_size, color=color, bgcolor=bgcolor)
    return img, name
