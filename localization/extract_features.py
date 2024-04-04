# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> extract_features.py
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 14:49
=================================================='''
import os
import os.path as osp
import h5py
import numpy as np
import progressbar
import yaml
import torch
import cv2
import torch.utils.data as Data
from tqdm import tqdm
from types import SimpleNamespace
import logging
import pprint
from pathlib import Path
import argparse
from nets.sfd2 import ResNet4x, extract_sfd2_return
from nets.superpoint import SuperPoint, extract_sp_return

confs = {
    'superpoint-n4096': {
        'output': 'feats-superpoint-n4096',
        'model': {
            'name': 'superpoint',
            'outdim': 256,
            'use_stability': False,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'conf_th': 0.005,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/superpoint_v1.pth"),
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': False,
        },
    },

    'resnet4x-20230511-210205-pho-0005': {
        'output': 'feats-resnet4x-20230511-210205-pho-0005',
        'model': {
            'outdim': 128,
            'name': 'resnet4x',
            'use_stability': False,
            'max_keypoints': 4096,
            'conf_th': 0.005,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/sfd2_20230511_210205_resnet4x.79.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': False,
        },
        'mask': False,
    },

    'sfd2': {
        'output': 'feats-sfd2',
        'model': {
            'outdim': 128,
            'name': 'resnet4x',
            'use_stability': False,
            'max_keypoints': 4096,
            'conf_th': 0.005,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/sfd2_20230511_210205_resnet4x.79.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': False,
        },
        'mask': False,
    },
}


class ImageDataset(Data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
    }

    def __init__(self, root, conf, image_list=None,
                 mask_root=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        if image_list is None:
            for g in conf.globs:
                self.paths += list(Path(root).glob('**/' + g))
            if len(self.paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            self.paths = [i.relative_to(root) for i in self.paths]
        else:
            with open(image_list, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    self.paths.append(Path(l))

        logging.info(f'Found {len(self.paths)} images in root {root}.')

        if mask_root is not None:
            self.mask_root = mask_root
        else:
            self.mask_root = None

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(w, h) > self.conf.resize_max):
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
        }

        if self.mask_root is not None:
            mask_path = Path(str(path).replace("jpg", "png"))
            if osp.exists(mask_path):
                mask = cv2.imread(str(self.mask_root / mask_path))
                mask = cv2.resize(mask, dsize=(image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros(shape=(image.shape[1], image.shape[2], 3), dtype=np.uint8)

            data['mask'] = mask

        return data

    def __len__(self):
        return len(self.paths)


def get_model(model_name, weight_path, outdim=128, **kwargs):
    if model_name == 'superpoint':
        model = SuperPoint(config={
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
            'weight_path': weight_path,
        }).eval()

        extractor = extract_sp_return

    if model_name == 'resnet4x':
        model = ResNet4x(outdim=outdim).eval()
        model.load_state_dict(torch.load(weight_path)['state_dict'], strict=True)
        extractor = extract_sfd2_return

    return model, extractor


@torch.no_grad()
def main(conf, image_dir, export_dir):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')
    model, extractor = get_model(model_name=conf['model']['name'], weight_path=conf["model"]["model_fn"],
                                 use_stability=conf['model']['use_stability'], outdim=conf['model']['outdim'])
    model = model.cuda()
    loader = ImageDataset(image_dir,
                          conf['preprocessing'],
                          image_list=args.image_list,
                          mask_root=None)
    loader = torch.utils.data.DataLoader(loader, num_workers=4)

    os.makedirs(export_dir, exist_ok=True)
    feature_path = Path(export_dir, conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    with tqdm(total=len(loader)) as t:
        for idx, data in enumerate(loader):
            t.update()
            pred = extractor(model, img=data["image"],
                             topK=conf["model"]["max_keypoints"],
                             mask=None,
                             conf_th=conf["model"]["conf_th"],
                             scales=conf["model"]["scales"],
                             )

            # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            pred['descriptors'] = pred['descriptors'].transpose()

            t.set_postfix(npoints=pred['keypoints'].shape[0])
            # print(pred['keypoints'].shape)

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            # pred['descriptors'] = pred['descriptors'].T
            if 'keypoints' in pred.keys():
                size = np.array(data['image'].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            grp = feature_file.create_group(data['name'][0])
            for k, v in pred.items():
                # print(k, v.shape)
                grp.create_dataset(k, data=v)

            del pred

    feature_file.close()
    logging.info('Finished exporting features.')

    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--image_list', type=str, default=None)
    parser.add_argument('--mask_dir', type=Path, default=None)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, required=True, choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir)
