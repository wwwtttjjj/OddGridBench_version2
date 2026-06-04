from PIL import Image
from pathlib import Path

# MIN_GRID = 3
# MAX_GRID = 5
# MIN_SET_SIZE = 12
# MAX_SET_SIZE = 16


#RAD, MPDD, GOODADS
MIN_GRID = 3
MAX_GRID = 4

MIN_SET_SIZE = 9
MAX_SET_SIZE = 12

MIN_IMG_MAX_SIDE = 350
MAX_IMG_MAX_SIDE = 600

MIN_GAP = 10
MAX_GAP = 20

MIN_MARGIN = 20
MAX_MARGIN = 35

MIN_CELL_PADDING = 20
MAX_CELL_PADDING = 35

BG_COLOR = (255, 255, 255)
MAX_CANVAS_SIZE = 2048





def resize_image_max_side(pil_img, max_size):
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img, 1.0
    scale = max_size / max(w, h)
    return pil_img.resize(
        (int(round(w * scale)), int(round(h * scale))),
        Image.BILINEAR,
    ), scale

def load_image_list(img_dir: Path):
    imgs = sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.bmp")) + sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")
    return imgs