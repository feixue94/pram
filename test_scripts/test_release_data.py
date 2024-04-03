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
    landmark_root = '/scratches/flyer_3/fx221/exp/localizer/resnet4x-20230511-210205-pho-0005-gm'
    # save_root = '/scratches/flyer_2/fx221/publications/pram_data/3D-models'
    save_root = '/scratches/flyer_2/fx221/publications/pram_data/landmarks'
    # dataset = '7Scenes'
    # dataset = '12Scenes'
    # dataset = 'CambridgeLandmarks'
    dataset = 'Aachen'
    config_file = 'configs/datasets/{:s}.yaml'.format(dataset)

    with open(config_file, 'rt') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    # print(configs)
    all_scenes = configs['scenes']
    for scene in all_scenes:
        scene_path = osp.join(landmark_root, dataset, scene)
        target_scene_path = osp.join(save_root, dataset, scene)
        os.makedirs(target_scene_path, exist_ok=True)

        n_cluster = configs[scene]['n_cluster']
        cluster_mode = configs[scene]['cluster_mode']
        cluster_method = configs[scene]['cluster_method']

        file_names = [
            'compress_model_birch',
            'point3D_desc.npy',
            'point3D_cluster_n{:d}_{:s}_birch.npy'.format(n_cluster, cluster_mode),
            'point3D_vrf_n{:d}_{:s}_birch.npy'.format(n_cluster, cluster_mode),
        ]

        for file_name in file_names:
            source_path = osp.join(scene_path, file_name)
            is_a_file = osp.isfile(source_path)
            is_a_dir = osp.isdir(source_path)
            if not is_a_file and not is_a_dir:
                print('{:s} not exist'.format(source_path))
                continue

            if is_a_file:
                shutil.copy2(source_path, osp.join(target_scene_path, file_name))
            else:
                shutil.copytree(source_path, osp.join(target_scene_path, file_name))

        ''' copying files for sfm '''
        '''
        file_names = [
            'pairs-db-covis20.txt',
            'pairs-query-netvlad20.txt',
            'queries_poses.txt',
            'queries_with_intrinsics.txt',
        ]
        for file_name in file_names:
            if not osp.isfile(osp.join(scene_path, file_name)):
                print('{:s} not exist'.format(osp.join(scene_path, file_name)))
                continue
            shutil.copy2(osp.join(scene_path, file_name), osp.join(target_scene_path, file_name))
        '''

        ''' copying 3D models'''
        '''
        sfm_path = 'sfm_resnet4x-20230511-210205-pho-0005-gm'
        raw_sfm_path = osp.join(origin_root, dataset, scene, sfm_path)

        save_path = osp.join(save_root, dataset, scene, '3D-models')
        os.makedirs(save_path, exist_ok=True)

        shutil.copy2(osp.join(raw_sfm_path, 'cameras.bin'), osp.join(save_path, 'cameras.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'images.bin'), osp.join(save_path, 'images.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'points3D.bin'), osp.join(save_path, 'points3D.bin'))
        shutil.copy2(osp.join(raw_sfm_path, 'statics.txt'), osp.join(save_path, 'statics.txt'))
        '''
