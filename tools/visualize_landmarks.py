# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> visualize_landmarks
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   22/03/2024 10:39
=================================================='''
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from colmap_utils.read_write_model import read_model, write_model, Point3D, Image, read_compressed_model
from recognition.vis_seg import generate_color_dic


def reconstruct_map(valid_image_ids, valid_p3d_ids, cameras, images, point3Ds, p3d_seg: dict):
    new_point3Ds = {}
    new_images = {}

    valid_p3d_ids_ = []
    for pid in tqdm(valid_p3d_ids, total=len(valid_p3d_ids)):

        if pid == -1:
            continue
        if pid not in point3Ds.keys():
            continue

        if pid not in p3d_seg.keys():
            continue

        sid = map_seg[pid]
        if sid == -1:
            continue
        valid_p3d_ids_.append(pid)

    valid_p3d_ids = valid_p3d_ids_
    print('valid_p3ds: ', len(valid_p3d_ids))

    # for im_id in tqdm(images.keys(), total=len(images.keys())):
    for im_id in tqdm(valid_image_ids, total=len(valid_image_ids)):
        im = images[im_id]
        # print('im: ', im)
        # exit(0)
        pids = im.point3D_ids
        valid_pids = []
        # for v in pids:
        #     if v not in valid_p3d_ids:
        #         valid_pids.append(-1)
        #     else:
        #         valid_pids.append(v)

        new_im = Image(id=im_id, qvec=im.qvec, tvec=im.tvec, camera_id=im.camera_id, name=im.name, xys=im.xys,
                       point3D_ids=pids)
        new_images[im_id] = new_im

    for pid in tqdm(valid_p3d_ids, total=len(valid_p3d_ids)):
        sid = map_seg[pid]

        xyz = points3D[pid].xyz
        if show_2D:
            xyz[1] = 0
            rgb = points3D[pid].rgb
        else:
            bgr = seg_color[sid + sid_start]
            rgb = np.array([bgr[2], bgr[1], bgr[0]])

        error = points3D[pid].error

        p3d = Point3D(id=pid, xyz=xyz, rgb=rgb, error=error,
                      image_ids=points3D[pid].image_ids,
                      point2D_idxs=points3D[pid].point2D_idxs)
        new_point3Ds[pid] = p3d

    return cameras, new_images, new_point3Ds


if __name__ == '__main__':
    save_root = '/scratches/flyer_3/fx221/exp/localizer/vis_clustering/'
    seg_color = generate_color_dic(n_seg=2000)
    data_root = '/scratches/flyer_3/fx221/exp/localizer/resnet4x-20230511-210205-pho-0005-gm'
    show_2D = False

    compress_map = False
    # compress_map = True

    # scene = 'Aachen/Aachenv11'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n512_xz_birch.npy'), allow_pickle=True)[()]
    # sid_start = 1
    # vrf_file_name = 'point3D_vrf_n512_xz_birch.npy'

    #
    # scene = 'CambridgeLandmarks/GreatCourt'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n32_xy_birch.npy'), allow_pickle=True)[()]
    # sid_start = 1

    # scene = 'CambridgeLandmarks/KingsCollege'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n32_xy_birch.npy'), allow_pickle=True)[()]
    # sid_start = 33
    # vrf_file_name = 'point3D_vrf_n32_xy_birch.npy'

    # scene = 'CambridgeLandmarks/StMarysChurch'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n32_xz_birch.npy'), allow_pickle=True)[()]
    # sid_start = 32 * 4 + 1
    # vrf_file_name = 'point3D_vrf_n32_xz_birch.npy'

    # scene = '7Scenes/office'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n16_xz_birch.npy'), allow_pickle=True)[()]
    # sid_start = 33

    # scene = '7Scenes/chess'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n16_xz_birch.npy'), allow_pickle=True)[()]
    # sid_start = 1
    # vrf_file_name = 'point3D_vrf_n16_xz_birch.npy'

    # scene = '7Scenes/redkitchen'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n16_xz_birch.npy'), allow_pickle=True)[()]
    # sid_start = 16 * 5 + 1
    # vrf_file_name = 'point3D_vrf_n16_xz_birch.npy'

    # scene = '12Scenes/apt1/kitchen'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n16_xy_birch.npy'), allow_pickle=True)[()]
    # sid_start = 1
    # vrf_file_name = 'point3D_vrf_n16_xy_birch.npy'

    # data_root = '/scratches/flyer_3/fx221/exp/localizer/resnet4x-20230511-210205-pho-0005-gml2'
    # scene = 'JesusCollege/jesuscollege'
    # seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n256_xy_birch.npy'), allow_pickle=True)[()]
    # sid_start = 1
    # vrf_file_name = 'point3D_vrf_n256_xy_birch.npy'

    scene = 'DarwinRGB/darwin'
    seg_data = np.load(osp.join(data_root, scene, 'point3D_cluster_n128_xy_birch.npy'), allow_pickle=True)[()]
    sid_start = 1
    vrf_file_name = 'point3D_vrf_n128_xy_birch.npy'

    cameras, images, points3D = read_model(osp.join(data_root, scene, 'model'), ext='.bin')
    print('Load {:d} 3D points from map'.format(len(points3D.keys())))

    if compress_map:
        vrf_data = np.load(osp.join(data_root, scene, vrf_file_name), allow_pickle=True)[()]
        valid_image_ids = [vrf_data[v][0]['image_id'] for v in vrf_data.keys()]
    else:
        valid_image_ids = list(images.keys())

    if compress_map:
        _, _, compress_points3D = read_compressed_model(osp.join(data_root, scene, 'compress_model_birch'),
                                                        ext='.bin')
        print('Load {:d} 3D points from compressed map'.format(len(compress_points3D.keys())))
        valid_p3d_ids = list(compress_points3D.keys())
    else:
        valid_p3d_ids = list(points3D.keys())

    save_path = osp.join(save_root, scene)

    if compress_map:
        save_path = save_path + '_comp'
    if show_2D:
        save_path = save_path + '_2D'

    os.makedirs(save_path, exist_ok=True)
    p3d_id = seg_data['id']
    seg_id = seg_data['label']
    map_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}

    new_cameras, new_images, new_point3Ds = reconstruct_map(valid_image_ids=valid_image_ids,
                                                            valid_p3d_ids=valid_p3d_ids, cameras=cameras, images=images,
                                                            point3Ds=points3D, p3d_seg=map_seg)

    # write_model(cameras=cameras, images=images, points3D=new_point3Ds,
    #             path=save_path, ext='.bin')
    write_model(cameras=new_cameras, images=new_images, points3D=new_point3Ds, path=save_path, ext='.bin')
