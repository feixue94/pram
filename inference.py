# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> inference
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/04/2024 16:06
=================================================='''
import argparse
import os
import os.path as osp
import torch
import torchvision.transforms.transforms as tvt
import yaml
from nets.segnet import SegNet
from nets.segnetvit import SegNetViT
from nets.sfd2 import load_sfd2

from dataset.get_dataset import compose_datasets
from tools.common import torch_set_gpu

torch.set_grad_enabled(True)

parser = argparse.ArgumentParser(description='PRAM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, required=True, help='config of specifications')
parser.add_argument('--landmark_path', type=str, required=True, help='path of landmarks')
parser.add_argument('--feat_weight_path', type=str, default='weights/sfd2_20230511_210205_resnet4x.79.pth')
parser.add_argument('--rec_weight_path', type=str, required=True, help='recognition weight')
parser.add_argument('--online', action='store_true', help='online visualization with pangolin')


def get_model(config):
    desc_dim = 256 if config['feature'] == 'spp' else 128
    if config['use_mid_feature']:
        desc_dim = 256
    model_config = {
        'network': {
            'descriptor_dim': desc_dim,
            'n_layers': config['layers'],
            'ac_fn': config['ac_fn'],
            'norm_fn': config['norm_fn'],
            'n_class': config['n_class'],
            'output_dim': config['output_dim'],
            'with_cls': config['with_cls'],
            'with_sc': config['with_sc'],
            'with_score': config['with_score'],
        }
    }

    if config['network'] == 'segnet':
        model = SegNet(model_config.get('network', {}))
        config['with_cls'] = False
    elif config['network'] == 'segnetvit':
        model = SegNetViT(model_config.get('network', {}))
        config['with_cls'] = False
    else:
        raise 'ERROR! {:s} model does not exist'.format(config['network'])

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['landmark_path'] = args.landmark_path
    torch_set_gpu(gpus=config['gpu'])

    feat_model = load_sfd2(weight_path=args.feat_weight_path).cuda().eval()
    print('Load SFD2 weight from {:s}'.format(args.feat_weight_path))

    rec_model = get_model(config=config)
    state_dict = torch.load(args.rec_weight_path, map_location='cpu')['model']
    rec_model.load_state_dict(state_dict, strict=True)
    print('Load recognition weight from {:s}'.format(args.rec_weight_path))

    img_transforms = []
    img_transforms.append(tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    img_transforms = tvt.Compose(img_transforms)

    dataset = config['dataset']
    if not args.online:
        from localization.loc_by_rec_eval import loc_by_rec_eval

        test_set = compose_datasets(datasets=dataset, config=config, train=False, sample_ratio=1)
        config['n_class'] = test_set.n_class

        loc_by_rec_eval(rec_model=rec_model.cuda().eval(),
                        loader=test_set,
                        local_feat=feat_model.cuda().eval(),
                        config=config, img_transforms=img_transforms)
    else:
        from localization.loc_by_rec_online import loc_by_rec_online

        loc_by_rec_online(rec_model=rec_model.cuda().eval(),
                          local_feat=feat_model.cuda().eval(),
                          config=config, img_transforms=img_transforms)
