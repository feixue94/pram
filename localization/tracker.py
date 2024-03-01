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
from localization.frame import Frame


class Tracker:
    def __init__(self, locMap, viewer, matcher):
        self.locMap = locMap
        self.viewer = viewer
        self.matcher = matcher

        self.lost = True

        self.curr_frame = None
        self.last_frame = None

    def track(self, frame: Frame):
        pass

    def match_frame(self, frame: Frame, reference_frame: Frame):
        pass
