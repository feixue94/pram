# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> tracker
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/02/2024 16:58
=================================================='''
import numpy as np
from localization.simglelocmap import SingleLocMap
from localization.multilocmap import MultiLocMap


class Tracker:
    def __init__(self, mmap):
        self.curr_qvec = None
        self.curr_tvec = None
        self.curr_ref_img_id = None
        self.curr_scene = None
        self.last_qvec = None
        self.last_tvec = None
        self.last_ref_img_id = None
        self.last_scene = None

        self.lost = True

    def update_status():
        pass
