# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> trainer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 15:04
=================================================='''
import datetime
import os
import os.path as osp
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

import shutil
import torch
from torch.autograd import Variable
from tools.common import save_args_yaml, merge_tags
from tools.metrics import compute_iou, compute_precision, SeqIOU, compute_corr_incorr, compute_seg_loss_weight
from tools.metrics import compute_cls_loss_ce, compute_cls_corr


class Trainer:
    def __init__(self, model, train_loader, feat_model=None, eval_loader=None, config=None, img_transforms=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.with_aug = self.config['with_aug']
        self.with_cls = self.config['with_cls']
        self.with_sc = self.config['with_sc']
        self.img_transforms = img_transforms
        self.feat_model = feat_model.cuda().eval() if feat_model is not None else None

        self.init_lr = self.config['lr']
        self.min_lr = self.config['min_lr']

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params=params, lr=self.init_lr)
        self.num_epochs = self.config['epochs']

        if config['resume_path'] is not None:
            log_dir = config['resume_path'].split('/')[-2]
            resume_log = torch.load(osp.join(osp.join(config['save_path'], config['resume_path'])), map_location='cpu')
            self.epoch = resume_log['epoch'] + 1
            if 'iteration' in resume_log.keys():
                self.iteration = resume_log['iteration']
            else:
                self.iteration = len(self.train_loader) * self.epoch
            self.min_loss = resume_log['min_loss']
        else:
            self.iteration = 0
            self.epoch = 0
            self.min_loss = 1e10

            now = datetime.datetime.now()
            all_tags = [now.strftime("%Y%m%d_%H%M%S")]
            dataset_name = merge_tags(self.config['dataset'], '')
            all_tags = all_tags + [self.config['network'], 'L' + str(self.config['layers']),
                                   dataset_name,
                                   str(self.config['feature']), 'B' + str(self.config['batch_size']),
                                   'K' + str(self.config['max_keypoints']), 'od' + str(self.config['output_dim']),
                                   'nc' + str(self.config['n_class'])]
            if self.config['use_mid_feature']:
                all_tags.append('md')
            if self.with_cls:
                all_tags.append(self.config['cls_loss'])
            if self.with_sc:
                all_tags.append(self.config['sc_loss'])
            if self.with_aug:
                all_tags.append('A')

            all_tags.append(self.config['cluster_method'])
            log_dir = merge_tags(tags=all_tags, connection='_')

        if config['local_rank'] == 0:
            self.save_dir = osp.join(self.config['save_path'], log_dir)
            os.makedirs(self.save_dir, exist_ok=True)

            print("save_dir: ", self.save_dir)

            self.log_file = open(osp.join(self.save_dir, "log.txt"), "a+")
            save_args_yaml(args=config, save_path=Path(self.save_dir, "args.yaml"))
            self.writer = SummaryWriter(self.save_dir)

            self.tag = log_dir

        self.do_eval = self.config['do_eval']
        if self.do_eval:
            self.eval_fun = None
            self.seq_metric = SeqIOU(n_class=self.config['n_class'], ignored_sids=[0])

    def preprocess_input(self, pred):
        for k in pred.keys():
            if k.find('name') >= 0:
                continue
            if k != 'image' and k != 'depth':
                if type(pred[k]) == torch.Tensor:
                    pred[k] = Variable(pred[k].float().cuda())
                else:
                    pred[k] = Variable(torch.stack(pred[k]).float().cuda())

        if self.with_aug:
            new_scores = []
            new_descs = []
            global_descs = []
            with torch.no_grad():
                for i, im in enumerate(pred['image']):
                    img = torch.from_numpy(im[0]).cuda().float().permute(2, 0, 1)
                    # img = self.img_transforms(img)[None]
                    if self.img_transforms is not None:
                        img = self.img_transforms(img)[None]
                    else:
                        img = img[None]
                    out = self.feat_model.extract_local_global(data={'image': img})
                    global_descs.append(out['global_descriptors'])

                    seg_scores, seg_descs = self.feat_model.sample(score_map=out['score_map'],
                                                                   semi_descs=out['mid_features'] if self.config[
                                                                       'use_mid_feature'] else out['desc_map'],
                                                                   kpts=pred['keypoints'][i],
                                                                   norm_desc=self.config['norm_desc'])  # [D, N]
                    new_scores.append(seg_scores[None])
                    new_descs.append(seg_descs[None])
            pred['global_descriptors'] = global_descs
            pred['scores'] = torch.cat(new_scores, dim=0)
            pred['seg_descriptors'] = torch.cat(new_descs, dim=0).permute(0, 2, 1)  # -> [B, N, D]

    def process_epoch(self):
        self.model.train()

        epoch_cls_losses = []
        epoch_seg_losses = []
        epoch_losses = []
        epoch_acc_corr = []
        epoch_acc_incorr = []
        epoch_cls_acc = []

        epoch_sc_losses = []

        for bidx, pred in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            self.preprocess_input(pred)
            if 0 <= self.config['its_per_epoch'] <= bidx:
                break

            data = self.model(pred)
            for k, v in pred.items():
                pred[k] = v
            pred = {**pred, **data}

            seg_loss = compute_seg_loss_weight(pred=pred['prediction'],
                                               target=pred['gt_seg'],
                                               background_id=0,
                                               weight_background=0.1)
            acc_corr, acc_incorr = compute_corr_incorr(pred=pred['prediction'],
                                                       target=pred['gt_seg'],
                                                       ignored_ids=[0])

            if self.with_cls:
                pred_cls_dist = pred['classification']
                gt_cls_dist = pred['gt_cls_dist']
                if len(pred_cls_dist.shape) > 2:
                    gt_cls_dist_full = gt_cls_dist.unsqueeze(-1).repeat(1, 1, pred_cls_dist.shape[-1])
                else:
                    gt_cls_dist_full = gt_cls_dist.unsqueeze(-1)
                cls_loss = compute_cls_loss_ce(pred=pred_cls_dist, target=gt_cls_dist_full)
                loss = seg_loss + cls_loss

                # gt_n_seg = pred['gt_n_seg']
                cls_acc = compute_cls_corr(pred=pred_cls_dist.squeeze(-1), target=gt_cls_dist)
            else:
                loss = seg_loss
                cls_loss = torch.zeros_like(seg_loss)
                cls_acc = torch.zeros_like(seg_loss)

            if self.with_sc:
                pass
            else:
                sc_loss = torch.zeros_like(seg_loss)

            epoch_losses.append(loss.item())
            epoch_seg_losses.append(seg_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_sc_losses.append(sc_loss.item())

            epoch_acc_corr.append(acc_corr.item())
            epoch_acc_incorr.append(acc_incorr.item())
            epoch_cls_acc.append(cls_acc.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iteration += 1

            lr = min(self.config['lr'] * self.config['decay_rate'] ** (self.iteration - self.config['decay_iter']),
                     self.config['lr'])
            if lr < self.min_lr:
                lr = self.min_lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.config['local_rank'] == 0 and bidx % self.config['log_intervals'] == 0:
                print_text = 'Epoch [{:d}/{:d}], Step [{:d}/{:d}/{:d}], Loss [s{:.2f}/c{:.2f}/sc{:.2f}/t{:.2f}], Acc [c{:.2f}/{:.2f}/{:.2f}]'.format(
                    self.epoch,
                    self.num_epochs, bidx,
                    len(self.train_loader),
                    self.iteration,
                    seg_loss.item(),
                    cls_loss.item(),
                    sc_loss.item(),
                    loss.item(),

                    np.mean(epoch_acc_corr),
                    np.mean(epoch_acc_incorr),
                    np.mean(epoch_cls_acc)
                )

                print(print_text)
                self.log_file.write(print_text + '\n')

                info = {
                    'lr': lr,
                    'loss': loss.item(),
                    'cls_loss': cls_loss.item(),
                    'sc_loss': sc_loss.item(),
                    'acc_corr': acc_corr.item(),
                    'acc_incorr': acc_incorr.item(),
                    'acc_cls': cls_acc.item(),
                }

                for k, v in info.items():
                    self.writer.add_scalar(tag=k, scalar_value=v, global_step=self.iteration)

        if self.config['local_rank'] == 0:
            print_text = 'Epoch [{:d}/{:d}], AVG Loss [s{:.2f}/c{:.2f}/sc{:.2f}/t{:.2f}], Acc [c{:.2f}/{:.2f}/{:.2f}]\n'.format(
                self.epoch,
                self.num_epochs,
                np.mean(epoch_seg_losses),
                np.mean(epoch_cls_losses),
                np.mean(epoch_sc_losses),
                np.mean(epoch_losses),
                np.mean(epoch_acc_corr),
                np.mean(epoch_acc_incorr),
                np.mean(epoch_cls_acc),
            )
            print(print_text)
            self.log_file.write(print_text + '\n')
            self.log_file.flush()
        return np.mean(epoch_losses)

    def eval_seg(self, loader):
        print('Start to do evaluation...')

        self.model.eval()
        self.seq_metric.clear()
        mean_iou_day = []
        mean_iou_night = []
        mean_prec_day = []
        mean_prec_night = []
        mean_cls_day = []
        mean_cls_night = []

        for bid, pred in tqdm(enumerate(loader), total=len(loader)):
            for k in pred.keys():
                if k.find('name') >= 0:
                    continue
                if k != 'image' and k != 'depth':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].float().cuda())
                    elif type(pred[k]) == np.ndarray:
                        pred[k] = Variable(torch.from_numpy(pred[k]).float()[None].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).float().cuda())

            if self.with_aug:
                with torch.no_grad():
                    if isinstance(pred['image'][0], list):
                        img = pred['image'][0][0]
                    else:
                        img = pred['image'][0]

                    img = torch.from_numpy(img).cuda().float().permute(2, 0, 1)
                    if self.img_transforms is not None:
                        img = self.img_transforms(img)[None]
                    else:
                        img = img[None]

                    encoder_out = self.feat_model.extract_local_global(data={'image': img})
                    global_descriptors = [encoder_out['global_descriptors']]
                    pred['global_descriptors'] = global_descriptors
                    if self.config['use_mid_feature']:
                        scores, descs = self.feat_model.sample(score_map=encoder_out['score_map'],
                                                               semi_descs=encoder_out['mid_features'],
                                                               kpts=pred['keypoints'][0],
                                                               norm_desc=self.config['norm_desc'])
                        # print('eval: ', scores.shape, descs.shape)
                        pred['scores'] = scores[None]
                        pred['seg_descriptors'] = descs[None].permute(0, 2, 1)  # -> [B, N, D]
                    else:
                        pred['seg_descriptors'] = pred['descriptors']

            image_name = pred['file_name'][0]
            with torch.no_grad():
                out = self.model(pred)
                pred = {**pred, **out}

                pred_seg = torch.max(pred['prediction'], dim=-1)[1]  # [B, N, C]
                pred_seg = pred_seg[0].cpu().numpy()
                gt_seg = pred['gt_seg'][0].cpu().numpy()
                iou = compute_iou(pred=pred_seg, target=gt_seg, n_class=self.config['n_class'], ignored_ids=[0])
                prec = compute_precision(pred=pred_seg, target=gt_seg, ignored_ids=[0])

                if self.with_cls:
                    pred_cls_dist = pred['classification']
                    gt_cls_dist = pred['gt_cls_dist']
                    cls_acc = compute_cls_corr(pred=pred_cls_dist.squeeze(-1), target=gt_cls_dist).item()
                else:
                    cls_acc = 0.

                if image_name.find('night') >= 0:
                    mean_iou_night.append(iou)
                    mean_prec_night.append(prec)
                    mean_cls_night.append(cls_acc)
                else:
                    mean_iou_day.append(iou)
                    mean_prec_day.append(prec)
                    mean_cls_day.append(cls_acc)

        print_txt = 'Eval Epoch {:d}, iou day/night {:.3f}/{:.3f}, prec day/night {:.3f}/{:.3f}, cls day/night {:.3f}/{:.3f}'.format(
            self.epoch, np.mean(mean_iou_day), np.mean(mean_iou_night),
            np.mean(mean_prec_day), np.mean(mean_prec_night),
            np.mean(mean_cls_day), np.mean(mean_cls_night))
        self.log_file.write(print_txt + '\n')
        print(print_txt)

        info = {
            'mean_iou_day': np.mean(mean_iou_day),
            'mean_iou_night': np.mean(mean_iou_night),
            'mean_prec_day': np.mean(mean_prec_day),
            'mean_prec_night': np.mean(mean_prec_night),
        }

        for k, v in info.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=self.epoch)

        return np.mean(mean_prec_night)

    def train(self):
        if self.config['local_rank'] == 0:
            print('Start to train the model from epoch: {:d}'.format(self.epoch))
            hist_values = []
            min_value = self.min_loss

        epoch = self.epoch
        while epoch < self.num_epochs:
            if self.config['with_dist']:
                self.train_loader.sampler.set_epoch(epoch=epoch)
            self.epoch = epoch

            train_loss = self.process_epoch()

            # return with loss INF/NAN
            if train_loss is None:
                continue

            if self.config['local_rank'] == 0:
                if self.do_eval and self.epoch % self.config['eval_n_epoch'] == 0:  # and self.epoch >= 50:
                    eval_ratio = self.eval_seg(loader=self.eval_loader)

                    hist_values.append(eval_ratio)  # higher better
                else:
                    hist_values.append(-train_loss)  # lower better

                checkpoint_path = os.path.join(self.save_dir,
                                               '%s.%02d.pth' % (self.config['network'], self.epoch))
                checkpoint = {
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'model': self.model.state_dict(),
                    'min_loss': min_value,
                }
                # for multi-gpu training
                if len(self.config['gpu']) > 1:
                    checkpoint['model'] = self.model.module.state_dict()

                torch.save(checkpoint, checkpoint_path)

                if hist_values[-1] < min_value:
                    min_value = hist_values[-1]
                    best_checkpoint_path = os.path.join(
                        self.save_dir,
                        '%s.best.pth' % (self.tag)
                    )
                    shutil.copy(checkpoint_path, best_checkpoint_path)
            # important!!!
            epoch += 1

        if self.config['local_rank'] == 0:
            self.log_file.close()
