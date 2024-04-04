#!/bin/bash
colmap=/home/mifs/fx221/Research/Code/thirdparty/colmap/build/src/exe/colmap
root_dir=/scratches/flyer_2

#feat=resnet4x-20230511-210205-pho-001
#feat=resnet4x-20230513-164306-pho-d64-001
feat=sfd2
matcher=gml

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=0
match_db=0
triangulation=1
localize=1


ransac_thresh=12
opt_thresh=12
covisibility_frame=20
inlier_thresh=30
radius=20
obs_thresh=3


#for scene in heads fire office stairs pumpkin redkitchen chess
#for scene in fire office pumpkin redkitchen chess
for scene in heads
do
  echo $scene
  dataset=/scratches/flyer_3/fx221/dataset/7Scenes/$scene
  image_dir=$dataset
  db_pair=$dataset/pairs-db-covis20.txt
  outputs=/scratches/flyer_2/fx221/localization/outputs/7Scenes/$scene
  query_pair=$dataset/pairs-query-netvlad20.txt
  gt_pose_fn=$dataset/queries_poses.txt
  save_root=$root_dir/fx221/exp/sgd2/7Scenes/$scene


  if [ "$extract_feat_db" -gt "0" ]; then
    python3 -m localization.extract_features --image_dir $dataset --export_dir $outputs/ --conf $feat
  fi

  if [ "$match_db" -gt "0" ]; then
    python3 -m localization.match_features --pairs $db_pair --export_dir $outputs/ --conf $matcher --features feats-$feat
  fi

  if [ "$triangulation" -gt "0" ]; then
    python3 -m localization.triangulation \
      --sfm_dir $outputs/sfm_$feat-$matcher \
      --reference_sfm_model $dataset/3D-models \
      --image_dir $dataset \
      --pairs $db_pair \
      --features $outputs/feats-$feat.h5 \
      --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5
  fi

  if [ "$localize" -gt "0" ]; then
    python3 -m localization.localizer \
      --dataset 7Scenes \
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