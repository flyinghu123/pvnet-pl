import os
import pytorch_lightning as pl
import torchvision
import random
import numpy as np
from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler  # SubsetRandomSampler
from .samplers import ImageSizeBatchSampler
from pycocotools.coco import COCO
import cv2
from typing import Optional
import time
from torch.utils.data.dataloader import default_collate
import torch
from omegaconf import DictConfig


linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp',
                     'BingMaYong', 'CQJXW', 'jiepu', 'lifan', 'YADIPro', 'Coffee', 'Tiguan', 'Wangzai']

class Compose(object):
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
class ToTensor(object):
    def __call__(self, img, kpts, mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, kpts, mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask

def read_linemod_mask(path, ann_type, cls_idx):
    if ann_type == 'real':
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:
            return (mask[..., 0] != 0).astype(np.uint8)
        else:
            return (mask != 0).astype(np.uint8)
    elif ann_type == 'fuse':
        return (np.asarray(Image.open(path)) == cls_idx).astype(np.uint8)
    elif ann_type == 'render':
        return (np.asarray(Image.open(path))).astype(np.uint8)
    
def compute_vertex(mask, kpt_2d, is_norm=True):
    h, w = mask.shape
    m = kpt_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]

    vertex = kpt_2d[None] - xy[:, None]
    if is_norm:
        norm = np.linalg.norm(vertex, axis=2, keepdims=True)
        norm[norm < 1e-3] += 1e-3
        vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    vertex_out = np.reshape(vertex_out, [h, w, m * 2])

    return vertex_out
def rotate_instance(img, mask, hcoords, rot_ang_min, rot_ang_max):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    hs, ws = np.nonzero(mask)
    R = cv2.getRotationMatrix2D((np.mean(ws), np.mean(hs)), degree, 1)
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    last_row = np.asarray([[0, 0, 1]], np.float32)
    hcoords = np.matmul(hcoords, np.concatenate([R, last_row], 0).transpose())
    return img, mask, hcoords
def crop_or_padding_to_fixed_size_instance(img, mask, hcoords, th, tw, overlap_ratio=0.5):
    h, w, _ = img.shape
    hs, ws = np.nonzero(mask)

    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)
    fh, fw = hmax - hmin, wmax - wmin
    hpad, wpad = th >= h, tw >= w

    hrmax = int(min(hmin + overlap_ratio * fh, h - th))  # h must > target_height else hrmax<0
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmin + overlap_ratio * fw, w - tw))  # w must > target_width else wrmax<0
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))

    # hbeg = 0 if hpad else np.random.randint(hrmin, hrmax)
    # hend = hbeg + th
    # wbeg = 0 if wpad else np.random.randint(wrmin,
    #                                         wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    # wend = wbeg + tw
    if hrmin < hrmax:
        hbeg = 0 if hpad else np.random.randint(hrmin, hrmax)
    else:
        hbeg = 0 if hpad else np.random.randint(hmin, h - th)
    hend = hbeg + th
    if wrmin < wrmax:
        wbeg = 0 if wpad else np.random.randint(wrmin, wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    else:
        wbeg = 0 if wpad else np.random.randint(wmin, w - tw)
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    hcoords[:, 0] -= wbeg * hcoords[:, 2]
    hcoords[:, 1] -= hbeg * hcoords[:, 2]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        hcoords[:, 0] += wbeg * hcoords[:, 2]
        hcoords[:, 1] += hbeg * hcoords[:, 2]

        img, mask = new_img, new_mask

    return img, mask, hcoords
def crop_resize_instance_v1(img, mask, hcoords, imheight, imwidth,
                            overlap_ratio=0.5, ratio_min=0.8, ratio_max=1.2):
    '''

    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap
    :param img:
    :param mask:
    :param hcoords:
    :param imheight:
    :param imwidth:
    :param overlap_ratio:
    :param ratio_min:
    :param ratio_max:
    :return:
    '''
    resize_ratio = np.random.uniform(ratio_min, ratio_max)
    target_height = int(imheight * resize_ratio)
    target_width = int(imwidth * resize_ratio)

    img, mask, hcoords = crop_or_padding_to_fixed_size_instance(
        img, mask, hcoords, target_height, target_width, overlap_ratio)

    img = cv2.resize(img, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)

    hcoords[:, 0] = hcoords[:, 0] / resize_ratio
    hcoords[:, 1] = hcoords[:, 1] / resize_ratio

    return img, mask, hcoords
def crop_or_padding_to_fixed_size(img, mask, th, tw):
    h, w, _ = img.shape
    hpad, wpad = th >= h, tw >= w

    hbeg = 0 if hpad else np.random.randint(0, h - th)
    wbeg = 0 if wpad else np.random.randint(0,
                                            w - tw)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    hend = hbeg + th
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask

        img, mask = new_img, new_mask

    return img, mask

def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


def pvnet_collator(batch):
    if 'pose_test' not in batch[0]['meta']:
        return default_collate(batch)

    inp = np.stack(batch[0]['inp'])
    inp = torch.from_numpy(inp)
    meta = default_collate([b['meta'] for b in batch])
    ret = {'inp': inp, 'meta': meta}

    return ret

class LinemodDataset(Dataset):
    def __init__(self, ann_file, data_root, split, transforms=None):
        self.data_root = data_root
        self.split = split
        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms


    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        # 拼接路径
        path = self.path_transfer(path)
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_cls_names.index(anno['cls']) + 1
        mask_path = anno['mask_path']
        mask_path = self.path_transfer(mask_path)
        mask = read_linemod_mask(mask_path, anno['type'], cls_idx)

        return inp, kpt_2d, mask

    def path_transfer(self, path):
        return os.path.join(self.data_root, path.replace('data/linemod/', ''))
    
    
    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask = self.read_data(img_id)
        img = np.asarray(img).astype(np.uint8)

        img_offset = np.array([0, 0])  # x, y
        # img_scale = np.array([1, 1])
        img_scale = 1

        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id,
               # 'img_offset': img_offset, 'img_scale': img_scale, 'kpt_2d': kpt_2d,
               'img_offset': img_offset, 'img_scale': img_scale, 'kpt_2d': kpt_2d.astype(np.float32), 'sym_num': 1,
               'meta': {}}


        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, -30, 30)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         0.8,
                                                         0.8,
                                                         1.2)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask

class LineModDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, data_dir: str = "data/linemod/"):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.train_transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test_transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.data_dir = data_dir
        self.train_ann = os.path.join(data_dir, cfg.data.cls_type, 'train.json')
        self.test_ann = os.path.join(data_dir, cfg.data.cls_type, 'test.json')
        # if subset_rate is None:
        #     self.train_sampler = RandomSampler(train_dataset)
        #     self.test_sampler = SequentialSampler(test_dataset)
        # else:
        #     self.train_sampler = RandomSubsetSampler(train_dataset, subset_rate)
        #     self.test_sampler = SequentialSampler(range(int(subset_rate * len(test_dataset))))
        # self.train_sampler = ImageSizeBatchSampler(self.train_sampler, batch_size, False, 256, 480, 640)
        # self.test_sampler = ImageSizeBatchSampler(self.test_sampler, batch_size, False, 256, 480, 640)
        
    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            pass
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

        if stage == "predict" or stage is None:
            pass

    def train_dataloader(self):
        train_dataset = LinemodDataset(self.train_ann, self.data_dir, 'train', transforms=self.train_transform)
        train_sampler = RandomSampler(train_dataset)
        train_sampler = ImageSizeBatchSampler(train_sampler, self.cfg.train.batch_size, False, 256, 480, 640)
        return DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)

    def val_dataloader(self):
        test_dataset = LinemodDataset(self.test_ann, self.data_dir, 'test', transforms=self.test_transform)
        test_sampler = SequentialSampler(test_dataset)
        test_sampler = ImageSizeBatchSampler(test_sampler, self.cfg.train.batch_size, False, 256, 480, 640)
        return DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=True)

    def test_dataloader(self):
        test_dataset = LinemodDataset(self.test_ann, self.data_dir, 'test', transforms=self.test_transform)
        test_sampler = SequentialSampler(test_dataset)
        test_sampler = ImageSizeBatchSampler(test_sampler, self.cfg.train.batch_size, False, 256, 480, 640)
        return DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=True)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), pin_memory=True)