# shapes/__init__.py
from .registry import draw_random_shape, draw_shape_by_name, shape_registry
from . import svg_shapes
from .svg_shapes import register_all_svg

__all__ = ["draw_random_shape", "draw_shape_by_name", "shape_registry", "register_all_svg"]
