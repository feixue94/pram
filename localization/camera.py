# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> camera
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 11:27
=================================================='''
import collections

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
