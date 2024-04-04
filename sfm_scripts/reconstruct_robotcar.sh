#!/bin/bash
colmap=/home/mifs/fx221/Research/Software/bin/colmap
root_dir=/scratches/flyer_2

dataset=/scratches/flyer_3/fx221/dataset/RobotCar-Seasons/RobotCar-Seasons
image_dir=$root_dir/fx221/localization/RobotCar-Seasons/images
outputs=$root_dir/fx221/localization/outputs/RobotCar-Seasons/RobotCar-Seasons
db_pair=$root_dir/fx221/localization/outputs/robotcar/pairs-db-covis20.txt
query_fn=$dataset/queries_with_intrinsics_rear.txt
query_pair=$dataset/pairs-query-netvlad20-percam-perloc-rear.txt
#gt_pose_fn=/data/cornucopia/fx221/localization/RobotCar-Seasons/3D-models/query_poses_v2.txt
gt_pose_fn=$dataset/queries_pose_spp_spg.txt
save_root=$root_dir/fx221/exp/sgd2/RobotCar-Seasons/RobotCar-Seasons


#feat=resnet4x-20230511-210205-pho-001
#feat=resnet4x-20230513-164306-pho-d64-001
feat=resnet4x-20230511-210205-pho-0005
#matcher=NNM
matcher=gm

#feat=superpoint-n4096
#matcher=superglue

extract_feat_db=1
match_db=1
triangulation=1
localize=1


if [ "$extract_feat_db" -gt "0" ]; then
  python3 -m loc.extract_features --image_dir $dataset/images --export_dir $outputs/ --conf $feat
fi


if [ "$match_db" -gt "0" ]; then
  python3 -m loc.match_features --pairs $outputs/pairs-db-covis20.txt --export_dir $outputs/ --conf $matcher --features feats-$feat
fi

if [ "$triangulation" -gt "0" ]; then
  python3 -m loc.triangulation \
    --sfm_dir $outputs/sfm_$feat-$matcher \
    --reference_sfm_model $dataset/sfm-sift \
    --image_dir $dataset/images/ \
    --pairs $outputs/pairs-db-covis20.txt \
    --features $outputs/feats-$feat.h5 \
    --matches $outputs/feats-$feat-$matcher-pairs-db-covis20.h5 \
    --colmap_path $colmap
fi

ransac_thresh=12
opt_thresh=12
covisibility_frame=20
inlier_thresh=100
radius=20
obs_thresh=3

# with opt
if [ "$localize" -gt "0" ]; then
  python3 -m loc.localizer \
    --dataset robotcar \
    --image_dir $image_dir \
    --save_root $save_root \
    --gt_pose_fn $gt_pose_fn \
    --retrieval $query_pair \
    --reference_sfm $outputs/sfm_$feat-$matcher \
    --queries $dataset/queries_with_intrinsics_rear.txt \
    --features $outputs/feats-$feat.h5 \
    --matcher_method $matcher \
    --ransac_thresh $ransac_thresh \
    --covisibility_frame $covisibility_frame \
    --radius $radius \
    --inlier_thresh $inlier_thresh \
    --obs_thresh $obs_thresh \
    --opt_thresh $opt_thresh \
    --use_hloc
fi

#