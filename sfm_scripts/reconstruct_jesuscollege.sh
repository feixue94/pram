#!/bin/bash

dataset_dir=/scratches/flyer_3/fx221/dataset/JesusCollege/jesuscollege
ref_sfm_dir=/scratches/flyer_3/fx221/dataset/JesusCollege/jesuscollege
output_dir=/scratches/flyer_2/fx221/localization/outputs/JesusCollege/jesuscollege

db_image_dir=$dataset_dir/db
query_image_dir=$dataset_dir/query
outputs=$output_dir
ref_sfm=$ref_sfm_dir/3D-models
db_pair=$ref_sfm_dir/pairs-db-covis20.txt
query_pair=$ref_sfm_dir/pairs-query-20.txt
query_fn=$ref_sfm_dir/queries_with_intrinsics.txt
gt_pose_fn=$ref_sfm_dir/queries_poses.txt

feat=sfd2
matcher=gm

extract_feat_db=1
match_db=1
triangulation=1
extract_feat_query=0
localize=0

if [ "$extract_feat_db" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $db_image_dir --export_dir $outputs/ --conf $feat
fi

if [ "$match_db" -gt "0" ]; then
  python3 -m loc.match_features --pairs $db_pair --export_dir $outputs/ --conf $matcher --features feats-$feat
fi

if [ "$triangulation" -gt "0" ]; then
  python3 -m loc.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $ref_sfm \
    --image_dir $db_image_dir \
    --pairs $db_pair \
    --features $outputs/feats-$feat.h5 \
    --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5
fi

if [ "$extract_feat_query" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $query_image_dir --export_dir $outputs/ --conf $feat
fi

ransac_thresh=15
opt_thresh=15
covisibility_frame=30
inlier_thresh=80
obs_thresh=3

if [ "$localize" -gt "0" ]; then
  python3 -m loc.localizer \
    --dataset aachen_v1.1 \
    --image_dir $query_image_dir \
    --save_root $outputs \
    --gt_pose_fn $gt_pose_fn \
    --retrieval $query_pair \
    --reference_sfm $outputs/sfm_$feat-$matcher \
    --queries $query_fn \
    --features $outputs/feats-$feat.h5 \
    --matcher_method $matcher \
    --ransac_thresh $ransac_thresh \
    --covisibility_frame $covisibility_frame \
    --obs_thresh $obs_thresh \
    --opt_thresh $opt_thresh \
    --inlier_thresh $inlier_thresh \
    --use_hloc
fi
