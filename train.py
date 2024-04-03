# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/04/2024 16:33
=================================================='''
import argparse
import os
import os.path as osp
import torch
import torchvision.transforms.transforms as tvt
import yaml
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.distributed as dist

from nets.sfd2 import load_sfd2
from nets.segnet import SegNet
from nets.segnetvit import SegNetViT
from dataset.utils import collect_batch
from dataset.get_dataset import compose_datasets
from tools.common import torch_set_gpu
from trainer import Trainer


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


parser = argparse.ArgumentParser(description='PRAM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, required=True, help='config of specifications')
parser.add_argument('--landmark_path', type=str, required=True, help='path of landmarks')
parser.add_argument('--feat_weight_path', type=str, default='weights/sfd2_20230511_210205_resnet4x.79.pth')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_DDP(rank, world_size, model, config, train_set, test_set, feat_model, img_transforms):
    print('In train_DDP..., rank: ', rank)
    torch.cuda.set_device(rank)

    device = torch.device(f'cuda:{rank}')
    if feat_model is not None:
        feat_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    setup(rank=rank, world_size=world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    shuffle=True,
                                                                    rank=rank,
                                                                    num_replicas=world_size,
                                                                    drop_last=True,  # important?
                                                                    )
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config['batch_size'] // world_size,
                                               num_workers=config['workers'] // world_size,
                                               # num_workers=1,
                                               pin_memory=True,
                                               # persistent_workers=True,
                                               shuffle=False,  # must be False
                                               drop_last=True,
                                               collate_fn=collect_batch,
                                               prefetch_factor=4,
                                               sampler=train_sampler)
    config['local_rank'] = rank

    if rank == 0:
        test_set = test_set
    else:
        test_set = None

    trainer = Trainer(model=model, train_loader=train_loader, feat_model=feat_model, eval_loader=test_set,
                      config=config, img_transforms=img_transforms)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    torch_set_gpu(gpus=config['gpu'])
    if config['local_rank'] == 0:
        print(config)

    img_transforms = []
    img_transforms.append(tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    img_transforms = tvt.Compose(img_transforms)

    feat_model = load_sfd2(weight_path=args.feat_weight_path).cuda().eval()
    print('Load SFD2 weight from {:s}'.format(args.feat_weight_path))

    dataset = config['dataset']
    train_set = compose_datasets(datasets=dataset, config=config, train=True, sample_ratio=None)
    if config['do_eval']:
        test_set = compose_datasets(datasets=dataset, config=config, train=False, sample_ratio=None)
    else:
        test_set = None
    config['n_class'] = train_set.n_class
    model = get_model(config=config)
    if config['local_rank'] == 0:
        if config['resume_path'] is not None:  # only for training
            model.load_state_dict(
                torch.load(osp.join(config['save_path'], config['resume_path']), map_location='cpu')['model'],
                strict=True)
            print('Load resume weight from {:s}'.format(osp.join(config['save_path'], config['resume_path'])))

    if not config['with_dist'] or len(config['gpu']) == 1:
        config['with_dist'] = False
        model = model.cuda()
        train_loader = Data.DataLoader(dataset=train_set,
                                       shuffle=True,
                                       batch_size=config['batch_size'],
                                       drop_last=True,
                                       collate_fn=collect_batch,
                                       num_workers=config['workers'])
        if test_set is not None:
            test_loader = Data.DataLoader(dataset=test_set,
                                          shuffle=False,
                                          batch_size=1,
                                          drop_last=False,
                                          collate_fn=collect_batch,
                                          num_workers=4)
        else:
            test_loader = None
        trainer = Trainer(model=model, train_loader=train_loader, feat_model=feat_model, eval_loader=test_loader,
                          config=config, img_transforms=img_transforms)
        trainer.train()
    else:
        mp.spawn(train_DDP, nprocs=len(config['gpu']),
                 args=(len(config['gpu']), model, config, train_set, test_set, feat_model, img_transforms),
                 join=True)
