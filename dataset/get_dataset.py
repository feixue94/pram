# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> get_dataset
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:40
=================================================='''
import os.path as osp
import yaml
from dataset.aachen import Aachen
from dataset.twelve_scenes import TwelveScenes
from dataset.seven_scenes import SevenScenes
from dataset.cambridge_landmarks import CambridgeLandmarks
from dataset.customdataset import CustomDataset


# from dataset.recdataset import RecDataset

def get_dataset(dataset):
    if dataset in ['7Scenes', 'S']:
        return SevenScenes
    elif dataset in ['12Scenes', 'T']:
        return TwelveScenes
    elif dataset in ['Aachen', 'A']:
        return Aachen
    elif dataset in ['CambridgeLandmarks', 'C']:
        return CambridgeLandmarks
    else:
        return CustomDataset


def compose_datasets(datasets, config, train=True, sample_ratio=None):
    sub_sets = []
    for name in datasets:
        if name == 'S':
            ds_name = '7Scenes'
        elif name == 'T':
            ds_name = '12Scenes'
        elif name == 'A':
            ds_name = 'Aachen'
        elif name == 'R':
            ds_name = 'RobotCar-Seasons'
        elif name == 'C':
            ds_name = 'CambridgeLandmarks'
        else:
            ds_name = name
            # raise '{} dataset does not exist'.format(name)
        segment_path = osp.join(config['segment_path'], ds_name)
        dataset_path = osp.join(config['dataset_path'], ds_name)
        scene_config_path = 'configs/datasets/{:s}.yaml'.format(ds_name)

        with open(scene_config_path, 'r') as f:
            scene_config = yaml.load(f, Loader=yaml.Loader)
        DSet = get_dataset(dataset=ds_name)

        for scene in scene_config['scenes']:
            if sample_ratio is None:
                scene_sample_ratio = scene_config[scene]['training_sample_ratio'] if train else scene_config[scene][
                    'eval_sample_ratio']
            else:
                scene_sample_ratio = sample_ratio
            scene_set = DSet(segment_path=segment_path,
                             dataset_path=dataset_path,
                             scene=scene,
                             seg_mode=scene_config[scene]['cluster_mode'],
                             seg_method=scene_config[scene]['cluster_method'],
                             n_class=scene_config[scene]['n_cluster'] + 1,  # including invalid - 0
                             dataset=ds_name,
                             train=train,
                             nfeatures=config['max_keypoints'] if train else config['eval_max_keypoints'],
                             min_inliers=config['min_inliers'],
                             max_inliers=config['max_inliers'],
                             random_inliers=config['random_inliers'],
                             with_aug=config['with_aug'],
                             jitter_params=config['jitter_params'],
                             scale_params=config['scale_params'],
                             image_dim=config['image_dim'],
                             query_p3d_fn=osp.join(config['segment_path'], ds_name, scene,
                                                   'point3D_query_n{:d}_{:s}_{:s}.npy'.format(
                                                       scene_config[scene]['n_cluster'],
                                                       scene_config[scene]['cluster_mode'],
                                                       scene_config[scene]['cluster_method'])),
                             query_info_path=osp.join(config['dataset_path'], ds_name, scene,
                                                      'queries_with_intrinsics.txt'),
                             sample_ratio=scene_sample_ratio,
                             )

            sub_sets.append(scene_set)

    return RecDataset(sub_sets=sub_sets)
