# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> test_release_data
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/04/2024 11:29
=================================================='''
import os
import os.path as osp
import yaml
import shutil

if __name__ == '__main__':
    origin_root = '/scratches/flyer_2/fx221/localization/outputs'
    dataset_root = '/scratches/flyer_3/fx221/dataset'
    save_root = '/scratches/flyer_2/fx221/publications/pram_data/3D-models'
    # dataset = '7Scenes'
    # dataset = '12Scenes'
    # dataset = 'CambridgeLandmarks'
    dataset = 'Aachen'
    config_file = 'configs/datasets/{:s}.yaml'.format(dataset)

    with open(config_file, 'rt') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    print(configs)
    all_scenes = configs['scenes']
    for scene in all_scenes:
        scene_path = osp.join(dataset_root, dataset, scene)
        target_scene_path = osp.join(save_root, dataset, scene)

        file_names = [
            'pairs-db-covis20.txt',
            'pairs-query-netvlad20.txt',
            'queries_poses.txt',
            'queries_with_intrinsics.txt',
        ]
        for file_name in file_names:
            shutil.copy2(osp.join(scene_path, file_name), osp.join(target_scene_path, file_name))

        ''' copying 3D models'''
        sfm_path = 'sfm_resnet4x-20230511-210205-pho-0005-gm'
        raw_sfm_path = osp.join(origin_root, dataset, scene, sfm_path)

        save_path = osp.join(save_root, dataset, scene, '3D-models')
        os.makedirs(save_path, exist_ok=True)

        shutil.copy2(osp.join(raw_sfm_path, 'cameras.bin'), osp.join(save_path, 'cameras.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'images.bin'), osp.join(save_path, 'images.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'points3D.bin'), osp.join(save_path, 'points3D.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'statics.txt'), osp.join(save_path, 'statics.txt'))
