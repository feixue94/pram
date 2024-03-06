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
from localization.frame import Frame


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

        self.scene = None
        self.current_vrf_id = None
        self.reference_frame_ids = None
        self.subMap = None
        self.seg_point_clouds = None
        self.point_clouds = None

        self.start_seg_id = 1
        self.stop = False

        self.refinement = False
        self.tracking = False

        # time
        self.time_feat = np.NAN
        self.time_rec = np.NAN
        self.time_loc = np.NAN
        self.time_ref = np.NAN

    def draw_3d_points_white(self):
        if self.point_clouds is None:
            return

        point_size = self.point_size * 0.5
        glColor4f(0.9, 0.95, 1.0, 0.6)
        glPointSize(point_size)
        pangolin.glDrawPoints(self.point_clouds)

    def draw_seg_3d_points(self):
        if self.seg_xyzs is None:
            return
        for sid in self.seg_xyzs.keys():
            xyzs = self.seg_xyzs[sid]
            point_size = self.point_size * 0.5
            bgr = self.seg_colors(sid + self.start_seg_id + 1)
            glColor3f(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
            glPointSize(point_size)
            pangolin.glDrawPoints(xyzs)

    def draw_ref_3d_points(self, use_seg_color=False):
        if self.reference_frame_ids is None:
            return
        ref_points3D_ids = []
        for fid in self.reference_frame_ids:
            pids = self.subMap.reference_frames[fid].points3D_ids()
            ref_points3D_ids.extend(list(pids))
        ref_points3D_ids = np.unique(ref_points3D_ids).tolist()

        point_size = self.point_size * 5
        glPointSize(point_size)
        glBegin(GL_POINTS)

        for pid in ref_points3D_ids:
            xyz = self.subMap.points3Ds[pid].xyz
            rgb = self.subMap.points3Ds[pid].rgb
            sid = self.subMap.points3Ds[pid].sid
            if use_seg_color:
                bgr = self.seg_colors(sid + self.start_seg_id + 1)
                glColor3f(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
            else:
                glColor3f(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

            glVertex3f(xyz[0], xyz[1], xyz[2])

        glEnd()

    def draw_vrf_frames(self):
        if self.subMap is None:
            return
        w = self.image_size * 1.0
        image_line_width = self.image_line_width * 1.0
        h = w * 0.75
        z = w * 0.6
        for sid in self.subMap.seg_vrfs.keys():
            frame_id = self.subMap.seg_vrfs[sid]
            qvec = self.subMap.reference_frames[frame_id].qvec
            tcw = self.subMap.reference_frames[frame_id].qvec

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

    def draw_current_vrf_frame(self):
        if self.current_vrf_id is None:
            return
        qvec = self.subMap.reference_frames[self.current_vrf_id].qvec
        tcw = self.subMap.reference_frames[self.current_vrf_id].tvec
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

    def draw_current_frame(self, Tcw, color=(0, 1.0, 0)):
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

    def draw_ref_frames(self):
        if self.reference_frame_ids is None:
            return
        w = self.image_size * 1.5
        image_line_width = self.image_line_width * 1.5
        h = w * 0.75
        z = w * 0.6
        for fid in self.reference_frame_ids:
            qvec = self.subMap.reference_frames[fid].qvec
            tcw = self.subMap.reference_frames[fid].tvec
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

    def terminate(self):
        lock = threading.Lock()
        lock.acquire()
        self.stop = True
        lock.release()

    def update_point_clouds(self):
        # for fast drawing
        seg_point_clouds = {}
        point_clouds = []
        for pid in self.subMap.points3Ds.keys():
            sid = self.subMap.points3Ds[pid].sid
            xyz = self.subMap.points3Ds[pid].xyz
            if sid in seg_point_clouds.keys():
                seg_point_clouds[sid].append(xyz.reshape(3, 1))
            else:
                seg_point_clouds[sid] = [xyz.reshape(3, 1)]

            point_clouds.append(xyz.reshape(3, 1))

        self.seg_point_clouds = seg_point_clouds
        self.point_clouds = point_clouds

    def update(self, curr_frame: Frame):
        lock = threading.Lock()
        lock.acquire()

        self.frame = curr_frame
        self.current_vrf_id = curr_frame.reference_frame_id
        self.subMap = self.locMap[curr_frame.matched_scene_name]
        if self.scene is None or self.scene != curr_frame.matched_scene_name:
            self.scene = curr_frame.matched_scene_name
            self.update_point_clouds()

        if curr_frame.qvec is not None:
            Rcw = qvec2rotmat(curr_frame.qvec)
            Tcw = np.column_stack((Rcw, curr_frame.tvec))
            self.Tcw = np.vstack((Tcw, (0, 0, 0, 1)))
            Rwc = Rcw.T
            twc = -Rcw.T @ curr_frame.tvec
            Twc = np.column_stack((Rwc, twc))
            self.Twc = np.vstack((Twc, (0, 0, 0, 1)))

        if curr_frame.gt_qvec is not None:
            gt_Rcw = qvec2rotmat(curr_frame.gt_qvec)
            gt_Tcw = np.column_stack((gt_Rcw, curr_frame.gt_tvec))
            self.gt_Tcw = np.vstack((gt_Tcw, (0, 0, 0, 1)))
            gt_Rwc = gt_Rcw.T
            gt_twc = -gt_Rcw.T @ curr_frame.gt_tvec
            gt_Twc = np.column_stack((gt_Rwc, gt_twc))
            self.gt_Twc = np.vstack((gt_Twc, (0, 0, 0, 1)))
        else:
            self.gt_Tcw = None
            self.gt_Twc = None

        # update time
        self.time_feat = curr_frame.time_feat
        self.time_rec = curr_frame.time_rec
        self.time_loc = curr_frame.time_loc
        self.time_ref = curr_frame.time_ref

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
                self.draw_3d_points_white()

            if menu.ShowRefPoints:
                self.draw_ref_3d_points(use_seg_color=False)
            if menu.ShowRefSegs:
                self.draw_ref_3d_points(use_seg_color=True)

            if menu.ShowSegs:
                self.draw_seg_3d_points()

            if menu.ShowAllVRFs:
                self.draw_vrf_frames()

            if menu.ShowRefFrames:
                self.draw_ref_frames()

            if menu.ShowVRFFrame:
                self.draw_current_vrf_frame()

            if menu.Refinement:
                self.refinement = True
            else:
                self.refinement = False

            if menu.Tracking:
                self.tracking = True
            else:
                self.tracking = False

            self.draw_current_frame(Tcw=self.Tcw)

            if self.gt_Tcw is not None:  # draw gt pose with color (0, 0, 1.0)
                self.draw_current_frame(Tcw=self.gt_Tcw, color=(0., 0., 1.0))

            d_img_rec.Activate()
            glColor4f(1, 1, 1, 1)

            # if self.image_rec is not None:
            #     d_img_rec_texture.Upload(self.image_rec, GL_RGB, GL_UNSIGNED_BYTE)
            #     d_img_rec_texture.RenderToViewportFlipY()

            time_total = 0
            if self.time_feat != np.NAN:
                menu.featTime = '{:.2f}s'.format(self.time_feat)
                time_total = time_total + self.time_feat
            if self.time_rec != np.NAN:
                menu.recTime = '{:.2f}s'.format(self.time_rec)
                time_total = time_total + self.time_rec
            if self.time_loc != np.NAN:
                menu.locTime = '{:.2f}s'.format(self.time_loc)
                time_total = time_total + self.time_loc
            if self.time_ref != np.NAN:
                menu.refTime = '{:.2f}s'.format(self.time_ref)
                time_total = time_total + self.time_ref
            menu.totalTime = '{:.2f}s'.format(time_total)

            time.sleep(50 / 1000)

            pangolin.FinishFrame()
