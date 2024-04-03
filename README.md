## Place Recognition Anywhere Model for Efficient Visual Localization

<p align="center">
  <img src="assets/overview.png" width="960">
</p>

Humans localize themselves efficiently in known environments by first recognizing landmarks defined on certain objects
and their spatial relationships, and then verifying the location by aligning detailed structures of recognized objects
with those in the memory. Inspired by this, we propose the place recognition anywhere model (PRAM) to perform visual
localization as efficiently as humans do. PRAM consists of two main components - recognition and registration. In
detail, first of all,
a self-supervised map-centric landmark definition strategy is adopted, making places in either indoor or outdoor scenes
act as unique landmarks. Then,
sparse keypoints extracted from images, are utilized as the input to a transformer-based deep neural network for
landmark recognition; these keypoints enable PRAM to recognize hundreds of landmarks with high time and memory
efficiency.
Keypoints along with recognized landmark labels are further used for registration between query images and the 3D
landmark map. Different
from previous hierarchical methods, PRAM discards global and local descriptors, and reduces over 90% storage. Since PRAM
utilizes
recognition and landmark-wise verification to replace global reference search and exhaustive matching respectively, it
runs 2.4 times
faster than prior state-of-the-art approaches. Moreover, PRAM opens new directions for visual localization including
multi-modality localization, map-centric feature learning, and hierarchical scene coordinate regression.

* Full paper PDF: [Place Recognition Anywhere Model for Efficient Visual Localization](assets/pram.pdf).

* Authors: *Fei Xue, Ignas Budvytis, Roberto Cipolla*

* Website: [PRAM](https://feixue94.github.io/pram-project) for videos, slides, recent updates, and datasets.

## Key Features

### 1. Self-supervised landmark definition on 3D space

- No need of segmentations on images
- No inconsistent semantic results from multi-view images
- No limitation to labels of only known objects
- Work in any places with known or unknown objects

### 2. Efficient landmark-wise coarse and fine localization

- Recognize landmarks as opposed to do global retrieval
- Local landmark-wise matching as opposed to exhaustive matching
- No global descriptors (e.g. NetVLAD)
- No reference images and their heavy repetative 2D keypoints and descriptors

### 3. Landmark-wise map sparsification

- Tack each landmark for localization
- Reduce redundant 3D points for each landmark independently

### 4. Sparse recognition

- Sparse SFD2 keypoints as tokens
- No uncertainties of points at boundaries
- Automatic inlier/outlier discrimination

### 5. Relocalization and temporal localization

- Per frame reclocalization from scratch
- Tracking previous frames for higher efficiency

### 6. One model one dataset

- All 7 subscenes in 7Scenes dataset share a model
- All 12 subscenes in 12Scenes dataset share a model
- All 5 subscenes in CambridgeLandmarks share a model

### 7. Flexibility to multi-modality input

- Any other signals (e.g. text, language, GPS, Magonemeter) can be used as tokens as input with SFD2 keypoints
- Joint task of localization and scene understanding

## Data preparation

1. Download the 7Scenes, 12Scenes, CambridgeLandmarks, and Aachen datasets
2. Do SfM with SFD2 features
3. Generate 3D landmarks from point clouds, create virtual reference frames and remove redundant 3D points

## Run the localization with online visualization

1. Download the [3D-models](https://drive.google.com/drive/folders/1Ws98KjWWKhWwyKMDgswa8-I4KJITS6Uw?usp=drive_link)
   obtained with SFD2
   keypoints, [pretrained models](https://drive.google.com/drive/folders/1Rlbo1MgSW9da27ZKLlSQF7t-Jli2fX36?usp=drive_link) ,
   and [landmarks](https://drive.google.com/drive/folders/1yT1ALo0L6kejPLmornUtODFUJLr9Q4M0?usp=drive_link)
2. Put these data in ```your_path``` and modify the path
3. Run the demo (e.g. 7Scenes)

```
python3 inference.py  --config configs/config_train_7scenes_resnet4x.yaml --rec_weight_path pretrained_models/7scenes_nc113_birch_segnetvit.199.pth  --landmark_path landmarks --online

```

```
python3 -m recognition.recmap
```

4. Train the recognition network for each dataset (e.g. 7Scenes)

```
python3 main.py --config configs/config_train_7scenes_resnet4x.yaml
```

5. Localization with online visualization (set loc: true, online: true)

```
python3 main.py --config configs/config_train_7scenes_resnet4x.yaml
```

## BibTeX Citation

If you use any ideas from the paper or code in this repo, please consider citing:

```
@inproceedings{xue2023sfd2,
  author    = {Fei Xue and Ignas Budvytis and Roberto Cipolla},
  title     = {SFD2: Semantic-guided Feature Detection and Description},
  booktitle = {CVPR},
  year      = {2023}
}

@inproceedings{xue2022imp,
  author    = {Fei Xue and Ignas Budvytis and Roberto Cipolla},
  title     = {IMP: Iterative Matching and Pose Estimation with Adaptive Pooling},
  booktitle = {CVPR},
  year      = {2023}
}

@inproceedings{xue2022efficient,
  author    = {Fei Xue and Ignas Budvytis and Daniel Olmeda Reino and Roberto Cipolla},
  title     = {Efficient Large-scale Localization by Global Instance Recognition},
  booktitle = {CVPR},
  year      = {2022}
}
```

## Acknowledgements

Part of the code is from previous excellent works
including [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
, [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
and [SGMNet](https://github.com/magicleap/SuperGluePretrainedNetwork). You can find more details from their released
repositories if you are interested in their works. 