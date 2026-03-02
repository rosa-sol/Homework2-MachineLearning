import os, re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

def _stem_to_breed(stem: str) -> str:
    # "american_bulldog_12" -> "american_bulldog"
    return re.sub(r"_\d+$", "", stem)

def _pseudo_bbox_from_image(rgb: np.ndarray):
    """
    Fast heuristic bbox generator (images-only).
    Returns (x1,y1,x2,y2) in pixel coords.
    """
    # grayscale
    gray = (0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]).astype(np.float32)
    # normalize
    g = (gray - gray.min()) / (np.ptp(gray) + 1e-6)

    # edge-ish map by simple gradient magnitude
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    edge = gx + gy

    # threshold: take top X% strongest edges
    thr = np.quantile(edge, 0.90)
    mask = edge >= thr

    ys, xs = np.where(mask)
    h, w = g.shape

    if ys.size < 50:
        # fallback: centered box
        x1 = int(w * 0.15)
        y1 = int(h * 0.15)
        x2 = int(w * 0.85)
        y2 = int(h * 0.85)
        return x1, y1, x2, y2

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # expand a bit
    pad_x = int(0.05 * (x2 - x1 + 1))
    pad_y = int(0.05 * (y2 - y1 + 1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    # sanity: ensure not tiny
    if (x2 - x1) < w * 0.2 or (y2 - y1) < h * 0.2:
        x1 = int(w * 0.15)
        y1 = int(h * 0.15)
        x2 = int(w * 0.85)
        y2 = int(h * 0.85)

    return int(x1), int(y1), int(x2), int(y2)

class OxfordPetsImagesOnlyDetectionSubset(Dataset):
    """
    Images-only Oxford Pets:
      - bbox is pseudo-generated from image (no annotations)
      - label = breed (5-class subset)
      - one object per image
    """
    def __init__(self, root, breeds, indices=None, img_size=384, cache=False):
        self.root = root
        self.img_size = img_size
        self.cache = cache
        self._cache = {}

        self.images_dir = os.path.join(root, "images")
        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Expected images folder at: {self.images_dir}")

        self.breeds = list(breeds)
        self.breed_to_label = {b: i + 1 for i, b in enumerate(self.breeds)}  # background=0

        all_imgs = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(".jpg")])
        items = []
        for fname in all_imgs:
            stem = os.path.splitext(fname)[0]
            breed = _stem_to_breed(stem)
            if breed in self.breed_to_label:
                items.append((fname, stem, breed))

        if len(items) == 0:
            raise RuntimeError("No matching images for the chosen 5 breeds. Check breed names vs filenames.")

        self.items = items
        self.indices = indices if indices is not None else list(range(len(self.items)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_i = self.indices[i]
        if self.cache and real_i in self._cache:
            return self._cache[real_i]

        fname, stem, breed = self.items[real_i]
        img_path = os.path.join(self.images_dir, fname)

        img = Image.open(img_path).convert("RGB")
        img = F.resize(img, [self.img_size, self.img_size])
        img_np = np.array(img, dtype=np.uint8)

        x1, y1, x2, y2 = _pseudo_bbox_from_image(img_np)

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([self.breed_to_label[breed]], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([real_i]),
        }

        out = (F.to_tensor(img), target)
        if self.cache:
            self._cache[real_i] = out
        return out

def collate_fn(batch):
    return tuple(zip(*batch))

