#!/bin/bash
colmap=/home/mifs/fx221/Research/Software/bin/colmap

root_dir=/scratches/flyer_2

dataset=/scratches/flyer_3/fx221/dataset/Aachen/Aachenv11
image_dir=$dataset/images/images_upright
outputs=/scratches/flyer_2/fx221/localization/outputs/Aachen/Aachenv11
query_pair=$dataset/pairs-query-netvlad50.txt
gt_pose_fn=$dataset/queries_pose_spp_spg.txt
save_root=$root_dir/fx221/exp/sgd2/Aachen/Aachenv11

#feat=resnet4x-20230511-210205-pho-001-r1600
#feat=resnet4x-20230513-164306-pho-d64-001-r1600
#matcher=NNM

feat=resnet4x-20230511-210205-pho-0005
matcher=gm

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=1
match_db=1
triangulation=1
localize=1

if [ "$extract_feat_db" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $dataset/images/images_upright --export_dir $outputs/ --conf $feat
fi

if [ "$match_db" -gt "0" ]; then
  python3 -m loc.match_features --pairs $dataset/pairs-db-covis20.txt --export_dir $outputs/ --conf $matcher --features feats-$feat
fi

if [ "$triangulation" -gt "0" ]; then
  python3 -m loc.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $dataset/3D-models \
    --image_dir $dataset/images/images_upright \
    --pairs $dataset/pairs-db-covis20.txt \
    --features $outputs/feats-$feat.h5 \
    --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5 \
    --colmap_path $colmap
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
    --image_dir $image_dir \
    --save_root $save_root \
    --gt_pose_fn $gt_pose_fn \
    --retrieval $query_pair \
    --reference_sfm $outputs/sfm_$feat-$matcher \
    --queries $dataset/queries/day_night_time_queries_with_intrinsics.txt \
    --features $outputs/feats-$feat.h5 \
    --matcher_method $matcher \
    --ransac_thresh $ransac_thresh \
    --covisibility_frame $covisibility_frame \
    --obs_thresh $obs_thresh \
    --opt_thresh $opt_thresh \
    --inlier_thresh $inlier_thresh \
    --use_hloc
fi