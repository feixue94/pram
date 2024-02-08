# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> loc_by_rec
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/02/2024 15:26
=================================================='''
import torch
from localization.multilocmap import MultiLocMap
import yaml, cv2, time
import os
import os.path as osp
import threading
from recognition.vis_seg import vis_seg_point, generate_color_dic
from tools.common import resize_img


