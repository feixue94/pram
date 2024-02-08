# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> visualizer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/02/2024 15:28
=================================================='''
import numpy as np
import pypangolin as pangolin
from OpenGL.GL import *
import yaml
import time
from copy import deepcopy
import threading
from collections import defaultdict
from colmap_utils.read_write_model import qvec2rotmat
from tools.common import resize_image_with_padding


class Visualizer:
    default_config = {
        'image_size_indoor': 0.05,
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

        self.points3D = None
        self.points3D_ref = None
        self.local_points3D = None

        self.images = None
        self.cameras = None

        self.map_seg = None

        # current camera pose
        self.Tcw = np.eye(4, dtype=float)
        self.Twc = np.linalg.inv(self.Tcw)
        self.gt_Tcw = None
        self.gt_Twc = None

        self.start_seg_id = 1
        self.mean_xyz = None
        self.image_rec = None
        self.image_loc = None

        self.pred_sid = None
        self.last_pred_sid = None
        self.pred_scene_name = None
        self.last_pred_scene_name = None
        self.reference_image_ids = None
        self.vrf_image_id = None
        self.rec_time = np.NAN
        self.loc_time = np.NAN
        self.ref_time = np.NAN
        self.total_time = np.NAN

        self.stop = False

        # options
        self.refinement = False
        self.point_clouds = None

    def draw_3d_point_white(self):
        if self.point_clouds is None:
            return

        point_size = self.point_size * 0.5
        glColor4f(0.9, 0.95, 1.0, 0.6)
        glPointSize(point_size)
        pangolin.glDrawPoints(self.point_clouds)

    def draw_ref_seg_3d_points(self):
        if self.reference_image_ids is None:
            return
        ref_pids = []
        for im in self.reference_image_ids:
            pids = self.images[im].point3D_ids
            for pid in pids:
                ref_pids.append(pid)

        point_size = self.point_size * 5
        glPointSize(point_size)
        glBegin(GL_POINTS)

        for pid in ref_pids:
            if pid == -1:
                continue
            if pid not in self.map_seg.keys():
                continue

            bgr = self.seg_colors[self.map_seg[pid] + self.start_seg_id + 1]
            glColor3f(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

            xyz = self.points3D[pid].xyz

            glVertex3f(xyz[0], xyz[1], xyz[2])

        glEnd()

    def draw_ref_3d_points(self):
        if self.reference_image_ids is None:
            return
        ref_pids = []
        for im in self.reference_image_ids:
            pids = self.images[im].point3D_ids
            for pid in pids:
                ref_pids.append(pid)

        point_size = self.point_size * 5
        glPointSize(point_size)
        glBegin(GL_POINTS)

        for pid in ref_pids:
            if pid == -1:
                continue
            if pid not in self.map_seg.keys():
                continue

            # bgr = self.seg_colors[self.map_seg[pid] + 1]

            xyz = self.points3D[pid].xyz
            rgb = self.points3D[pid].rgb
            glColor3f(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

            glVertex3f(xyz[0], xyz[1], xyz[2])

        glEnd()

    def draw_seg_3d_points(self):
        if self.map_seg is None:
            return
        point_size = self.point_size
        glPointSize(point_size)
        glBegin(GL_POINTS)

        # could be very slow due to per-point rendering
        for pid in self.map_seg.keys():
            if pid not in self.points3D.keys():
                continue
            xyz = self.points3D[pid].xyz
            sid = self.map_seg[pid]
            bgr = self.seg_colors[sid + 1]  # [bgr]
            glColor3f(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

            glVertex3f(xyz[0], xyz[1], xyz[2])
        glEnd()

    def draw_images(self, poses):
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

    def draw_vrf_images(self):
        if self.seg_vrf is None:
            return

        w = self.image_size * 1.0
        image_line_width = self.image_line_width * 1.0
        h = w * 0.75
        z = w * 0.6
        for sid in sorted(self.seg_vrf.keys()):
            qvec = self.seg_vrf[sid][0]['qvec']
            tcw = self.seg_vrf[sid][0]['tvec']

            Rcw = qvec2rotmat(qvec)

            twc = -Rcw.T @ tcw
            Rwc = Rcw.T

            Twc = np.column_stack((Rwc, twc))
            Twc = np.vstack((Twc, (0, 0, 0, 1)))

            glPushMatrix()

            glMultMatrixf(Twc.T)

            glLineWidth(image_line_width)
            glColor3f(1, 0, 0)
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

    def draw_ref_images(self):
        if self.reference_image_ids is None:
            return

        w = self.image_size * 1.5
        image_line_width = self.image_line_width * 1.5
        h = w * 0.75
        z = w * 0.6
        for im in sorted(self.reference_image_ids):
            qvec = self.images[im].qvec
            tcw = self.images[im].tvec

            Rcw = qvec2rotmat(qvec)

            twc = -Rcw.T @ tcw
            Rwc = Rcw.T

            Twc = np.column_stack((Rwc, twc))
            Twc = np.vstack((Twc, (0, 0, 0, 1)))

            glPushMatrix()

            glMultMatrixf(Twc.T)

            glLineWidth(image_line_width)
            glColor3f(100 / 255, 140 / 255, 17 / 255)
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

    def draw_current_image(self, Tcw, color=(0, 1.0, 0)):

        Twc = np.linalg.inv(Tcw)

        camera_line_width = self.image_line_width * 2
        w = self.image_size * 2
        h = w * 0.75
        z = w * 0.6

        glPushMatrix()

        glMultMatrixf(Twc.T)  # not the .T

        glLineWidth(camera_line_width)
        glColor3f(color[0], color[1], color[2])
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

    def draw_current_vrf_image(self):
        if self.vrf_image_id is None:
            return

        qvec = self.images[self.vrf_image_id].qvec
        tcw = self.images[self.vrf_image_id].tvec

        Rcw = qvec2rotmat(qvec)
        twc = -Rcw.T @ tcw
        Rwc = Rcw.T
        Twc = np.column_stack((Rwc, twc))
        Twc = np.vstack((Twc, (0, 0, 0, 1)))

        camera_line_width = self.image_line_width * 2
        w = self.image_size * 2
        h = w * 0.75
        z = w * 0.6

        glPushMatrix()

        glMultMatrixf(Twc.T)  # note the .T

        glLineWidth(camera_line_width)
        glColor3f(1, 0, 0)
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

    def update_current_image(self, qcw, tcw, gt_qcw=None, gt_tcw=None):
        lock = threading.Lock()
        lock.acquire()

        Rcw = qvec2rotmat(qcw)
        Tcw = np.column_stack((Rcw, tcw))
        self.Tcw = np.vstack((Tcw, (0, 0, 0, 1)))
        Rwc = Rcw.T
        twc = -Rcw.T @ tcw
        Twc = np.column_stack((Rwc, twc))
        self.Twc = np.vstack((Twc, (0, 0, 0, 1)))

        if gt_qcw is not None and gt_tcw is not None:
            gt_Rcw = qvec2rotmat(gt_qcw)
            gt_Tcw = np.column_stack((gt_Rcw, gt_tcw))
            self.gt_Tcw = np.vstack((gt_Tcw, (0, 0, 0, 1)))
            gt_Rwc = gt_Rcw.T
            gt_twc = -gt_Rcw.T @ gt_tcw
            gt_Twc = np.column_stack((gt_Rwc, gt_twc))
            self.gt_Twc = np.vstack((gt_Twc, (0, 0, 0, 1)))
        else:
            self.gt_Tcw = None
            self.gt_Twc = None

        lock.release()

    def update_rec_image(self, img):
        lock = threading.Lock()
        lock.acquire()
        if isinstance(img, list):
            one_img = [resize_image_with_padding(im, nw=self.img_width * 2, nh=self.img_height) for im in img]
            one_img = np.vstack(one_img)
        else:
            one_img = img
        self.image_rec = resize_image_with_padding(image=one_img, nw=self.img_width * 2, nh=self.img_height * 2)

        lock.release()

    def update_loc_image(self, img):
        lock = threading.Lock()
        lock.acquire()
        self.image_loc = resize_image_with_padding(image=img, nw=self.img_width * 2, nh=self.img_height,
                                                   padding_color=(0, 0, 0))
        lock.release()

    def update_loc_status(self, loc_info: dict):
        lock = threading.Lock()
        lock.acquire()
        self.last_pred_sid = self.pred_sid
        self.pred_sid = loc_info['pred_sid']
        self.vrf_image_id = loc_info['vrf_image_id']
        self.reference_image_ids = loc_info['reference_db_ids']
        lock.release()

        self.pred_scene_name = self.locMap.sid_scene_name[self.pred_sid]
        self.start_seg_id = self.locMap.scene_name_start_sid[self.pred_scene_name]
        self.pred_sid_in_sub_scene = self.pred_sid - self.locMap.scene_name_start_sid[self.pred_scene_name]
        self.pred_sub_map = self.locMap.sub_maps[self.pred_scene_name]
        self.pred_image_path_prefix = self.pred_sub_map.image_path_prefix

        if self.pred_scene_name != self.last_pred_scene_name:  # not the same submap
            self.images = self.pred_sub_map.map_images
            self.cameras = self.pred_sub_map.map_cameras
            self.points3D = self.pred_sub_map.map_p3ds
            self.map_seg = self.pred_sub_map.map_seg
            self.seg_map = self.pred_sub_map.seg_map
            self.seg_vrf = self.pred_sub_map.seg_vrf

            self.update_point_cloud()

    def update_point_cloud(self):
        point_clouds = []
        if self.local_points3D is None:
            vis_point3Ds = self.points3D
        else:
            vis_point3Ds = self.local_points3D
        for pid in vis_point3Ds:
            if pid == -1:
                continue
            xyz = self.points3D[pid].xyz
            point_clouds.append(xyz.reshape(3, 1))

        self.point_clouds = point_clouds

    def update_rec_time(self, t):
        lock = threading.Lock()
        lock.acquire()
        self.rec_time = t

        lock.release()

    def update_loc_time(self, t):
        lock = threading.Lock()
        lock.acquire()
        self.loc_time = t

        lock.release()

    def update_ref_time(self, t):
        lock = threading.Lock()
        lock.acquire()
        self.ref_time = t

        lock.release()

    def terminate(self):
        lock = threading.Lock()
        lock.acquire()
        self.stop = True
        lock.release()

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
        menu.FollowCamera = (True, pangolin.VarMeta(toggle=True))
        menu.ShowPoints = (False, pangolin.VarMeta(toggle=True))
        menu.ShowSegs = (False, pangolin.VarMeta(toggle=True))
        menu.ShowRefSegs = (True, pangolin.VarMeta(toggle=True))
        menu.ShowRefPoints = (False, pangolin.VarMeta(toggle=True))
        menu.ShowAllVRFs = (False, pangolin.VarMeta(toggle=True))
        menu.ShowRefFrames = (False, pangolin.VarMeta(toggle=True))
        menu.ShowVRFFrame = (True, pangolin.VarMeta(toggle=True))

        menu.Refinement = (self.refinement, pangolin.VarMeta(toggle=True))

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

            self.draw_current_image(Tcw=self.Tcw)
            if self.gt_Tcw is not None:  # draw gt pose with color (0, 0, 1.0)
                self.draw_current_image(Tcw=self.gt_Tcw, color=(0., 0., 1.0))

            d_img_rec.Activate()
            glColor4f(1, 1, 1, 1)

            if self.image_rec is not None:
                d_img_rec_texture.Upload(self.image_rec, GL_RGB, GL_UNSIGNED_BYTE)
                d_img_rec_texture.RenderToViewportFlipY()

            if self.rec_time != np.NAN:
                menu.recTime = '{:.2f}s'.format(self.rec_time)
            if self.loc_time != np.NAN:
                menu.locTime = '{:.2f}s'.format(self.loc_time)
            if self.ref_time != np.NAN:
                menu.refTime = '{:.2f}s'.format(self.ref_time)
            if self.rec_time != np.NAN and self.loc_time != np.NAN and self.ref_time != np.NAN:
                self.total_time = self.rec_time + self.loc_time + self.ref_time
                menu.totalTime = '{:.2f}s'.format(self.total_time)

            time.sleep(50 / 1000)

            pangolin.FinishFrame()
