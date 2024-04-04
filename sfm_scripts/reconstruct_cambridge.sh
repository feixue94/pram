#!/bin/bash

# you need to use your own path
dataset_dir=/scratches/flyer_3/fx221/dataset/CambridgeLandmarks
ref_sfm_dir=/scratches/flyer_2/fx221/publications/pram_data/3D-models/CambridgeLandmarks
output_dir=/scratches/flyer_2/fx221/localization/outputs/CambridgeLandmarks


feat=sfd2
matcher=gm

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=0
match_db=0
triangulation=0
localize=1

ransac_thresh=12
opt_thresh=12
covisibility_frame=20
inlier_thresh=30
radius=30
obs_thresh=3


#for scene in GreatCourt ShopFacade KingsCollege OldHospital StMarysChurch
for scene in StMarysChurch
#for scene in GreatCourt ShopFacade
do
  echo $scene

  image_dir=$dataset_dir/$scene
  ref_sfm=$ref_sfm_dir/$scene/3D-models
  db_pair=$ref_sfm_dir/$scene/pairs-db-covis20.txt
  outputs=$output_dir/$scene
  query_pair=$ref_sfm_dir/$scene/pairs-query-netvlad20.txt
  gt_pose_fn=$ref_sfm_dir/$scene/queries_poses.txt
  query_fn=$ref_sfm_dir/$scene/queries_with_intrinsics.txt

  if [ "$extract_feat_db" -gt "0" ]; then
    python3 -m loc.extract_features --image_dir $image_dir --export_dir $outputs/ --conf $feat
  fi

  if [ "$match_db" -gt "0" ]; then
    python3 -m loc.match_features --pairs $db_pair --export_dir $outputs/ --conf $matcher --features feats-$feat
  fi

  if [ "$triangulation" -gt "0" ]; then
    python3 -m loc.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $ref_sfm \
    --image_dir $image_dir\
    --pairs $db_pair \
    --features $outputs/feats-$feat.h5 \
    --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5
  fi

  if [ "$localize" -gt "0" ]; then
    python3 -m loc.localizer \
      --dataset cambridge \
      --image_dir $image_dir \
      --save_root $outputs\
      --gt_pose_fn $gt_pose_fn \
      --retrieval $query_pair \
      --reference_sfm $outputs/sfm_$feat-$matcher \
      --queries $query_fn \
      --features $outputs/feats-$feat.h5 \
      --matcher_method adagm2 \
      --ransac_thresh $ransac_thresh \
      --covisibility_frame $covisibility_frame \
      --obs_thresh $obs_thresh \
      --opt_thresh $opt_thresh \
      --inlier_thresh $inlier_thresh \
      --use_hloc
  fi

done