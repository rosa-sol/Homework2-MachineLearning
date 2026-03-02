import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class PennFudanDataset(Dataset):
    def __init__(self, root, indices=None, img_size=512, cache=False):
        self.root = root
        self.img_size = img_size

        #  add these two lines
        self.cache = cache
        self._cache = {}

        self.img_dir = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")

        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(".png")])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.lower().endswith(".png")])
        assert len(self.imgs) == len(self.masks)

        self.indices = indices if indices is not None else list(range(len(self.imgs)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # add this caching block
        if self.cache and idx in self._cache:
            return self._cache[idx]

        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        img = F.resize(img, [self.img_size, self.img_size])
        mask = Image.fromarray(mask).resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        boxes = []
        for oid in obj_ids:
            ys, xs = np.where(mask == oid)
            if ys.size == 0:
                continue
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        out = (F.to_tensor(img), target)

        #  store in cache
        if self.cache:
            self._cache[idx] = out

        return out

def collate_fn(batch):
    return tuple(zip(*batch))
