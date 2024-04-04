#!/bin/bash
#colmap=/home/mifs/fx221/Research/Code/thirdparty/colmap/build/src/exe/colmap
colmap=/home/mifs/fx221/Research/Code/thirdparty/colmap/build-kyuban/src/exe/colmap
root_dir=/scratches/flyer_2
weight_netvlad=/scratches/flyer_2/fx221/Research/Code/third_weights/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar

name=CUED

feat=resnet4x-20230511-210205-pho-0005
matcher=gm

extract_feat_db=0
match_db=0
triangulation=0
extract_global_feat=0
localize=1

ransac_thresh=12
opt_thresh=12
covisibility_frame=20
inlier_thresh=30
radius=20
obs_thresh=3


for scene in baker
do
  echo $scene
  dataset=/scratches/flyer_3/fx221/dataset/$name/$scene
  image_dir=$dataset
  db_pair=$dataset/pairs-db-covis20.txt
  outputs=/scratches/flyer_2/fx221/localization/outputs/$name/$scene
  query_pair=$dataset/pairs-query-netvlad20.txt
  gt_pose_fn=$dataset/queries_poses.txt
  save_root=$root_dir/fx221/exp/sgd2/$name/$scene


  if [ "$extract_feat_db" -gt "0" ]; then
    python3 -m loc.extract_features --image_dir $dataset --export_dir $outputs/ --conf $feat
  fi

  if [ "$match_db" -gt "0" ]; then
    python3 -m loc.match_features --pairs $db_pair --export_dir $outputs/ --conf $matcher --features feats-$feat
  fi

  if [ "$triangulation" -gt "0" ]; then
  python3 -m loc.triangulation \
      --sfm_dir $outputs/sfm_$feat-$matcher \
      --reference_sfm_model $dataset/3D-models \
      --image_dir $dataset \
      --pairs $db_pair \
      --features $outputs/feats-$feat.h5 \
      --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5 \
      --colmap_path $colmap
  fi

    if [ "$extract_global_feat" -gt "0" ]; then
  python3 -m loc.extract_global_features \
      --sfm_dir $outputs/sfm_$feat-$matcher \
      --image_dir $dataset \
      --query_image_list $dataset/queries_with_intrinsics.txt \
      --query_pair $query_pair \
      --features $outputs/netvlad.h5py \
      --weight $weight_netvlad \
      --topk 20
  fi

  if [ "$localize" -gt "0" ]; then
    python3 -m loc.localizer \
      --dataset CUED \
      --image_dir $image_dir \
      --save_root $save_root \
      --gt_pose_fn $gt_pose_fn \
      --retrieval $query_pair \
      --reference_sfm $outputs/sfm_$feat-$matcher \
      --queries $dataset/queries_with_intrinsics.txt \
      --features $outputs/feats-$feat.h5 \
      --matcher_method $matcher \
      --ransac_thresh $ransac_thresh \
      --covisibility_frame $covisibility_frame \
      --radius $radius \
      --obs_thresh $obs_thresh \
      --opt_thresh $opt_thresh \
      --inlier_thresh $inlier_thresh \
      --use_hloc
  fi
done
