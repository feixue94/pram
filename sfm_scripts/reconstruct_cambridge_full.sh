#!/bin/bash
# you need to use your own path
dataset_dir=/scratches/flyer_3/fx221/exp/camdata/cambridge_full/multi_slam/dump_data/db_10s
ref_sfm_dir=/scratches/flyer_3/fx221/exp/camdata/cambridge_full/multi_slam/dump_data/db_10s
output_dir=/scratches/flyer_3/fx221/exp/camdata/cambridge_full/multi_slam/dump_data/db_10s/output

# fixed
outputs=$output_dir
ref_sfm=$ref_sfm_dir/3D-models-empty
db_pair=$ref_sfm_dir/pairs-db20-offset1.txt

query_pair=$ref_sfm_dir/pairs-query-netvlad50.txt
gt_pose_fn=$ref_sfm_dir/queries_pose_spp_spg.txt
query_fn=$ref_sfm_dir/queries_with_intrinsics.txt



feat=sfd2
matcher=gml

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=0
match_db=1
triangulation=1
localize=0

if [ "$extract_feat_db" -gt "0" ]; then
  python3 -m localization.extract_features --image_dir $dataset_dir/ --export_dir $output_dir/ --conf $feat
fi

if [ "$match_db" -gt "0" ]; then
  python3 -m localization.match_features --pairs $db_pair --export_dir $output_dir/ --conf $matcher --features feats-$feat
fi

if [ "$triangulation" -gt "0" ]; then
  python3 -m localization.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $ref_sfm \
    --image_dir $dataset_dir/ \
    --pairs $db_pair \
    --features $output_dir/feats-$feat.h5 \
    --matches $output_dir/feats-$feat-$matcher-pairs-db20-offset1.h5
fi

ransac_thresh=15
opt_thresh=15
covisibility_frame=30
inlier_thresh=80
obs_thresh=3

if [ "$localize" -gt "0" ]; then
  python3 -m localization.localizer \
    --dataset aachen_v1.1 \
    --image_dir $image_dir \
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
