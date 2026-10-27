import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from . import BaseActor
from ..loss.cos_sim_loss import cosine_similarity_loss
from ...utils.box_ops import box_xywh_to_xyxy, box_cxcywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
import torchvision.transforms as transforms


class lightTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # triple loss
        # self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.triple = nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

        # self.transform = transforms.RandomErasing(p=0.05, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0, inplace=False)

        # validation visualization: save the first `val_vis_num` samples of every validation pass
        self.val_vis_num = getattr(cfg.TRAIN, 'VAL_VIS_NUM', 100) if cfg is not None else 100
        self.val_vis_dir = os.path.join(settings.env.workspace_dir, 'visualization', 'val')
        self._val_vis_epoch = None
        self._val_vis_count = 0

    def __call__(self, data):

        out_dict = self.forward_pass(data)

        loss, status = self.compute_losses(out_dict, data)

        # net.training is False while the validation loader is being cycled
        if not self.net.training and self.val_vis_num > 0 and self.settings.local_rank in [-1, 0]:
            self._save_val_visualizations(out_dict, data)

        return loss, status

    @torch.no_grad()
    def _save_val_visualizations(self, pred_dict, data):
        """Draw GT (green) and predicted (red) boxes on validation search images and save them to
        <workspace_dir>/visualization/val. Only the first `val_vis_num` samples of each validation epoch are saved."""
        epoch = data['epoch']
        if epoch != self._val_vis_epoch:  # new validation pass -> reset the counter
            self._val_vis_epoch = epoch
            self._val_vis_count = 0
        if self._val_vis_count >= self.val_vis_num:
            return

        os.makedirs(self.val_vis_dir, exist_ok=True)

        search_imgs = data['search_images'][:, 0, :].detach().float().cpu()  # (B,3,H,W), normalized
        gt_boxes = data['search_anno'].detach().float().cpu().view(-1, 4).numpy()  # (B,4) normalized xywh
        pred_boxes = pred_dict['pred_boxes'].detach().float().cpu().view(-1, 4)  # (B,4) normalized cxcywh
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes).numpy()  # normalized xyxy

        mean = np.array(self.cfg.DATA.MEAN, dtype=np.float32)
        std = np.array(self.cfg.DATA.STD, dtype=np.float32)

        for b in range(search_imgs.shape[0]):
            if self._val_vis_count >= self.val_vis_num:
                break

            # de-normalize -> uint8 BGR
            img = search_imgs[b].permute(1, 2, 0).numpy() * std + mean
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            scale = np.array([w, h, w, h], dtype=np.float32)

            gx, gy, gw, gh = gt_boxes[b]
            gt_xyxy = np.array([gx, gy, gx + gw, gy + gh], dtype=np.float32)
            pr_xyxy = pred_boxes[b]

            # IoU between GT and prediction (normalized coords are fine, IoU is scale-invariant per axis)
            iw = max(0.0, min(gt_xyxy[2], pr_xyxy[2]) - max(gt_xyxy[0], pr_xyxy[0]))
            ih = max(0.0, min(gt_xyxy[3], pr_xyxy[3]) - max(gt_xyxy[1], pr_xyxy[1]))
            inter = iw * ih
            union = gw * gh + max(0.0, pr_xyxy[2] - pr_xyxy[0]) * max(0.0, pr_xyxy[3] - pr_xyxy[1]) - inter
            iou = inter / union if union > 0 else 0.0

            gt_px = (gt_xyxy * scale).astype(int)
            pr_px = (pr_xyxy * scale).astype(int)
            cv2.rectangle(img, (gt_px[0], gt_px[1]), (gt_px[2], gt_px[3]), (0, 255, 0), 2)  # GT: green
            cv2.rectangle(img, (pr_px[0], pr_px[1]), (pr_px[2], pr_px[3]), (0, 0, 255), 2)  # pred: red
            cv2.putText(img, f'IoU {iou:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imwrite(os.path.join(self.val_vis_dir, f'ep{epoch:03d}_{self._val_vis_count:03d}.png'), img)
            self._val_vis_count += 1

    def forward_pass(self, data):
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][:, i, :].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][:, 0, :].view(-1, *data['search_images'].shape[2:])
        # search_img = self.transform(search_img)

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(z=template_list, x=search_img)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        bs, n, _ = gt_dict['search_anno'].shape
        gt_bbox = gt_dict['search_anno'].view(bs, 4)

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'].view(n, bs, 4), (480, 640),
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps_flatten = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # locate box
        try:
            iou_loss, iou = self.objective.iou(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            iou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # l1 loss
        l1_loss = self.objective.l1(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if 'score_map' in pred_dict:
            location_loss = self.objective.focal_loss(pred_dict['score_map'], gt_gaussian_maps_flatten)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)



        # weighted sum
        loss = self.loss_weight['iou'] * iou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss  # + compute_tri_loss * 0.05
        # * location_loss  # cos_sim_loss * 0.1

        # return
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": iou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      # "Loss/cossim": cos_sim_loss.item(),
                      # "Loss/triple": compute_tri_loss.item(),
                      "mean_IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss