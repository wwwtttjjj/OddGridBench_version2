# from torchvision import datasets, transforms

# mnist_train = datasets.MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )

# mnist_test = datasets.MNIST(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )

import numpy as np

def load_mnist_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

images = load_mnist_images("/data/wengtengjin/OddGridBench_version2/mnist/data/MNIST/raw/train-images-idx3-ubyte")
labels = load_mnist_labels("/data/wengtengjin/OddGridBench_version2/mnist/data/MNIST/raw/train-labels-idx1-ubyte")

import os
from PIL import Image

out_root = "mnist_png/train"
os.makedirs(out_root, exist_ok=True)

for i, (img, label) in enumerate(zip(images, labels)):
    label_dir = os.path.join(out_root, str(label))
    os.makedirs(label_dir, exist_ok=True)

    im = Image.fromarray(img, mode="L")
    im.save(os.path.join(label_dir, f"{i}.png"))