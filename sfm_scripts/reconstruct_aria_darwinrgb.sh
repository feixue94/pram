#!/bin/bash
colmap=/home/mifs/fx221/Research/Code/thirdparty/colmap/build_2204/src/exe/colmap

root_dir=/scratches/flyer_2

dataset=/scratches/flyer_3/fx221/dataset/DarwinRGB/darwin
db_image_dir=$dataset/db
query_image_dir=$dataset/query
outputs=/scratches/flyer_2/fx221/localization/outputs/DarwinRGB/darwin
query_pair=$dataset/pairs-query-20.txt
gt_pose_fn=$dataset/queries_poses.txt
save_root=$root_dir/fx221/exp/sgd2/DarwinRGB/darwin

feat=resnet4x-20230511-210205-pho-0005
matcher=gm

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=0
match_db=0
triangulation=0
extract_feat_query=0
localize=1

if [ "$extract_feat_db" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $db_image_dir --export_dir $outputs/ --conf $feat
fi

if [ "$match_db" -gt "0" ]; then
  python3 -m loc.match_features --pairs $dataset/pairs-db-covis20.txt --export_dir $outputs/ --conf $matcher --features feats-$feat
fi

if [ "$triangulation" -gt "0" ]; then
  python3 -m loc.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $dataset/3D-models \
    --image_dir $db_image_dir \
    --pairs $dataset/pairs-db-covis20.txt \
    --features $outputs/feats-$feat.h5 \
    --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5 \
    --colmap_path $colmap
fi

if [ "$extract_feat_query" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $query_image_dir --export_dir $outputs/ --conf $feat
fi

ransac_thresh=15
opt_thresh=15
covisibility_frame=30
inlier_thresh=80
radius=30
obs_thresh=3

if [ "$localize" -gt "0" ]; then
  python3 -m loc.localizer \
    --dataset aachen_v1.1 \
    --image_dir $query_image_dir \
    --save_root $save_root \
    --gt_pose_fn $gt_pose_fn \
    --retrieval $query_pair \
    --reference_sfm $outputs/sfm_$feat-$matcher \
    --queries $dataset/queries_with_intrinsics.txt \
    --features $outputs/feats-$feat.h5 \
    --matcher_method $matcher \
    --ransac_thresh $ransac_thresh \
    --covisibility_frame $covisibility_frame \
    --obs_thresh $obs_thresh \
    --opt_thresh $opt_thresh \
    --inlier_thresh $inlier_thresh \
    --use_hloc

fi
