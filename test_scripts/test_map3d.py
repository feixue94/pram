# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> test_map3d
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 11:13
=================================================='''
import numpy as np
import os
import os.path as osp
import yaml
from copy import deepcopy
from localization.singlemap3d import SingleMap3D

if __name__ == '__main__':
    scenes = []
    sid_scene_name = []
    sub_maps = {}
    scene_name_start_sid = {}

    config_path = 'configs/config_train_aachen_sfd2.yaml'
    with open(config_path, 'rt') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    n_class = 0
    datasets = ['Aachen']

    for name in datasets:
        config_path = osp.join(config['config_path'], '{:s}.yaml'.format(name))
        dataset_name = name

        with open(config_path, 'r') as f:
            scene_config = yaml.load(f, Loader=yaml.Loader)

        scenes = scene_config['scenes']
        for sid, scene in enumerate(scenes):
            scenes.append(name + '/' + scene)

            new_config = deepcopy(config)
            new_config['dataset_path'] = osp.join(config['dataset_path'], dataset_name, scene)
            new_config['segment_path'] = osp.join(config['segment_path'], dataset_name, scene)
            new_config['n_cluster'] = scene_config[scene]['n_cluster']
            new_config['cluster_mode'] = scene_config[scene]['cluster_mode']
            new_config['cluster_method'] = scene_config[scene]['cluster_method']
            new_config['gt_pose_path'] = scene_config[scene]['gt_pose_path']
            new_config['image_path_prefix'] = scene_config[scene]['image_path_prefix']
            sub_map = SingleMap3D(config=new_config,
                                  # with_compress=config['localization']['with_compress'],
                                  with_compress=False,
                                  )
            sub_maps[dataset_name + '/' + scene] = sub_map

            n_scene_class = scene_config[scene]['n_cluster']
            sid_scene_name = sid_scene_name + [dataset_name + '/' + scene for ni in range(n_scene_class)]
            scene_name_start_sid[dataset_name + '/' + scene] = n_class
            n_class = n_class + n_scene_class
    print('Load {} sub_maps from {} datasets'.format(len(sub_maps), len(datasets)))
