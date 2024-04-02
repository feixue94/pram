## Place Recognition Anywhere Model for Efficient Visual Localization

<p align="center">
  <img src="assets/overview1.png" width="960">
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

* Website: [PRAM](https://feixue94.github.io/) for videos, slides, recent updates, and datasets.

## Key Features

### Self-supervised landmark definition on 3D space

- No need of segmentation on images
- No inconsistent semantic labels from multi-view images
- No limitation of only known objects
- Work in any places with known or unknown objects

### Efficient landmark-wise coarse and fine localization

- Recognize landmarks as opposed to do global retrieval
- Local landmark-wise matching as opposed to exhaustive matching
- No global descriptors (e.g. NetVLAD)
- No reference images and their repetative 2D keypoints and descriptors

### Landmark-wise map sparsification

- Tack reference frame assigned to each landmark
- Reduce redundant 3D points for each landmark independently

### Sparse recognition

- Sparse SFD2 keypoints as tokens
- No uncertainties of points at boundaries
- Automatic inlier/outlier discrimination

### Relocalization and temporal localization

- Per frame reclocalization from scratch
- Tracking previous frames for higher efficiency

### Flexibility to multi-modality data

- Any other signals (e.g. text, language, GPS, Magonemeter) can be used as tokens as input with SFD2 keypoints
- Joint task of localization and scene understanding

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

@inproceedings{xue2022imp,
author = {Fei Xue and Ignas Budvytis and Roberto Cipolla},
title = {IMP: Iterative Matching and Pose Estimation with Adaptive Pooling},
booktitle = {CVPR},
year = {2023}
}

```

## Acknowledgements

Part of the code is from previous excellent works
including [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
, [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
and [SGMNet](https://github.com/magicleap/SuperGluePretrainedNetwork). You can find more details from their released
repositories if you are interested in their works. 