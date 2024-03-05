# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> viewer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   05/03/2024 16:50
=================================================='''
import numpy as np
import pypangolin as pangolin
from OpenGL.GL import *
import time
import threading
from colmap_utils.read_write_model import qvec2rotmat
from tools.common import resize_image_with_padding


class Viewer:
    default_config = {
        'image_size_indoor': 0.1,
        'image_line_width_indoor': 1,

        'image_size_outdoor': 1,
        'image_line_width_outdoor': 3,

        'point_size_indoor': 1,
        'point_size_outdoor': 1,

        'image_width': 640,
        'image_height': 480,

        'viewpoint_x': 0,
        'viewpoint_y': -1,
        'viewpoint_z': -5,
        'viewpoint_F': 512,

        'scene': 'indoor',
    }

    def __init__(self, locMap, seg_color, config={}):
        self.config = {**self.default_config, **config}
        self.viewpoint_x = self.config['viewpoint_x']
        self.viewpoint_y = self.config['viewpoint_y']
        self.viewpoint_z = self.config['viewpoint_z']
        self.viewpoint_F = self.config['viewpoint_F']
        self.img_width = self.config['image_width']
        self.img_height = self.config['image_height']

        if self.config['scene'] == 'indoor':
            self.image_size = self.config['image_size_indoor']
            self.image_line_width = self.config['image_line_width_indoor']
            self.point_size = self.config['point_size_indoor']

        else:
            self.image_size = self.config['image_size_outdoor']
            self.image_line_width = self.config['image_line_width_outdoor']
            self.point_size = self.config['point_size_outdoor']
            self.viewpoint_z = -150

        self.locMap = locMap
        self.seg_colors = seg_color

        # current camera pose
        self.frame = None
        self.Tcw = np.eye(4, dtype=float)
        self.Twc = np.linalg.inv(self.Tcw)
        self.gt_Tcw = None
        self.gt_Twc = None

        self.start_seg_id = 1
        self.stop = False

        self.refinement = False
        self.tracking = False
        self.point_clouds = None

    def draw_3d_points_white(self):
        if self.point_clouds is None:
            return

        point_size = self.point_size * 0.5
        glColor4f(0.9, 0.95, 1.0, 0.6)
        glPointSize(point_size)
        pangolin.glDrawPoints(self.point_clouds)

    def draw_frames(self, poses: list):
        w = self.image_size
        image_line_width = self.image_line_width
        h = w * 0.75
        z = w * 0.6
        for (qvec, tcw) in poses:
            Rcw = qvec2rotmat(qvec)
            twc = -Rcw.T @ tcw
            Rwc = Rcw.T

            Twc = np.column_stack((Rwc, twc))
            Twc = np.vstack((Twc, (0, 0, 0, 1)))

            glPushMatrix()

            glMultMatrixf(Twc.T)

            glLineWidth(image_line_width)
            glColor3f(0, 0, 1)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(w, h, z)
            glVertex3f(0, 0, 0)
            glVertex3f(w, -h, z)
            glVertex3f(0, 0, 0)
            glVertex3f(-w, -h, z)
            glVertex3f(0, 0, 0)
            glVertex3f(-w, h, z)

            glVertex3f(w, h, z)
            glVertex3f(w, -h, z)

            glVertex3f(-w, h, z)
            glVertex3f(-w, -h, z)

            glVertex3f(-w, h, z)
            glVertex3f(w, h, z)

            glVertex3f(-w, -h, z)
            glVertex3f(w, -h, z)
            glEnd()

            glPopMatrix()

    def run(self):
        pangolin.CreateWindowAndBind("Map reviewer", 640, 480)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pangolin.CreatePanel("menu").SetBounds(pangolin.Attach(0),
                                               pangolin.Attach(1),
                                               pangolin.Attach(0),
                                               # pangolin.Attach.Pix(-175),
                                               pangolin.Attach.Pix(175),
                                               # pangolin.Attach(1)
                                               )

        menu = pangolin.Var("menu")
        menu.Tracking = (False, pangolin.VarMeta(toggle=True))
        menu.FollowCamera = (True, pangolin.VarMeta(toggle=True))
        menu.ShowPoints = (False, pangolin.VarMeta(toggle=True))
        menu.ShowSegs = (False, pangolin.VarMeta(toggle=True))
        menu.ShowRefSegs = (True, pangolin.VarMeta(toggle=True))
        menu.ShowRefPoints = (False, pangolin.VarMeta(toggle=True))
        menu.ShowAllVRFs = (False, pangolin.VarMeta(toggle=True))
        menu.ShowRefFrames = (False, pangolin.VarMeta(toggle=True))
        menu.ShowVRFFrame = (True, pangolin.VarMeta(toggle=True))

        menu.Refinement = (self.refinement, pangolin.VarMeta(toggle=True))

        menu.featTime = 'NaN'
        menu.recTime = 'NaN'
        menu.locTime = 'NaN'
        menu.refTime = 'NaN'
        menu.totalTime = 'NaN'

        pm = pangolin.ProjectionMatrix(640, 480, self.viewpoint_F, self.viewpoint_F, 320, 240, 0.1,
                                       10000)

        # /camera position，viewpoint position，axis direction
        mv = pangolin.ModelViewLookAt(self.viewpoint_x,
                                      self.viewpoint_y,
                                      self.viewpoint_z,
                                      0, 0, 0,
                                      # 0.0, -1.0, 0.0,
                                      pangolin.AxisZ,
                                      )

        s_cam = pangolin.OpenGlRenderState(pm, mv)
        # Attach bottom, Attach top, Attach left, Attach right,
        scale = 0.42
        d_img_rec = pangolin.Display('image_rec').SetBounds(pangolin.Attach(1 - scale),
                                                            pangolin.Attach(1),
                                                            pangolin.Attach(
                                                                1 - 0.3),
                                                            pangolin.Attach(1),
                                                            self.img_width / self.img_height
                                                            )  # .SetLock(0, 1)

        handler = pangolin.Handler3D(s_cam)

        d_cam = pangolin.Display('3D').SetBounds(
            pangolin.Attach(0),  # bottom
            pangolin.Attach(1),  # top
            pangolin.Attach.Pix(175),  # left
            # pangolin.Attach.Pix(0),  # left
            pangolin.Attach(1),  # right
            -640 / 480,  # aspect
        ).SetHandler(handler)

        d_img_rec_texture = pangolin.GlTexture(self.img_width * 2, self.img_height * 2, GL_RGB, False, 0, GL_RGB,
                                               GL_UNSIGNED_BYTE)
        while not pangolin.ShouldQuit() and not self.stop:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # glClearColor(1.0, 1.0, 1.0, 1.0)
            glClearColor(0.0, 0.0, 0.0, 1.0)

            d_cam.Activate(s_cam)
            if menu.FollowCamera:
                s_cam.Follow(pangolin.OpenGlMatrix(self.Twc.astype(np.float32)), follow=True)

            # pangolin.glDrawColouredCube()
            if menu.ShowPoints:
                self.draw_3d_point_white()

            if menu.ShowRefPoints:
                self.draw_ref_3d_points()

            if menu.ShowSegs:
                self.draw_seg_3d_points()

            if menu.ShowRefSegs:
                self.draw_ref_seg_3d_points()

            if menu.ShowAllVRFs:
                self.draw_vrf_images()

            if menu.ShowRefFrames:
                self.draw_ref_images()

            if menu.ShowVRFFrame:
                self.draw_current_vrf_image()

            if menu.Refinement:
                self.refinement = True
            else:
                self.refinement = False

            if menu.Tracking:
                self.tracking = True
            else:
                self.tracking = False

            self.draw_current_image(Tcw=self.Tcw)

            if self.gt_Tcw is not None:  # draw gt pose with color (0, 0, 1.0)
                self.draw_current_image(Tcw=self.gt_Tcw, color=(0., 0., 1.0))

            d_img_rec.Activate()
            glColor4f(1, 1, 1, 1)

            if self.image_rec is not None:
                d_img_rec_texture.Upload(self.image_rec, GL_RGB, GL_UNSIGNED_BYTE)
                d_img_rec_texture.RenderToViewportFlipY()

            if self.feat_time != np.NAN:
                menu.featTime = '{:.2f}s'.format(self.feat_time)
            if self.rec_time != np.NAN:
                menu.recTime = '{:.2f}s'.format(self.rec_time)
            if self.loc_time != np.NAN:
                menu.locTime = '{:.2f}s'.format(self.loc_time)
            if self.ref_time != np.NAN:
                menu.refTime = '{:.2f}s'.format(self.ref_time)
            if self.rec_time != np.NAN and self.loc_time != np.NAN and self.ref_time != np.NAN:
                self.total_time = self.feat_time + self.rec_time + self.loc_time + self.ref_time
                menu.totalTime = '{:.2f}s'.format(self.total_time)

            time.sleep(50 / 1000)

            pangolin.FinishFrame()
