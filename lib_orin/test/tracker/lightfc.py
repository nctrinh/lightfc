import torch
import cv2 as cv
import os
import numpy as np

from lib.models import LightFC
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_iou, box_xyxy_to_xywh
from lib.test.utils.hann import hann2d
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
import lib.train.data.processing_utils as prutils


class lightFC(BaseTracker):
    def __init__(self, params, dataset_name):
        super(lightFC, self).__init__(params)

        network = LightFC(cfg=params.cfg, env_num=None, training=False)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        checkpoint_path = '/mnt/tupm/LightFC/checkpoints/train/lightfc/building_new_model/LightFC_best.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
        network.load_state_dict(state_dict)
        network.to('cuda')
        network.eval()

        for module in network.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in network.fusion.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in network.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.visualize_dir = '/mnt/tupm/LightFC/output/visualization'
        self.visualize_num = 500

        self.feat_sz = (30, 40)

        # motion constrain
        # self.output_window = hann2d(torch.tensor([self.feat_sz[0], self.feat_sz[1]]).long(), centered=True).cuda()

        self.frame_id = 0

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def initialize(self, image, info: dict):
        H, W, _ = image.shape

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        save_template = cv.cvtColor(z_patch_arr, cv.COLOR_RGB2BGR)
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
        cv.imwrite(os.path.join(self.visualize_dir, 'template.png'), save_template)

        with torch.no_grad():
            self.z_feat = self.network.forward_backbone(template.tensors)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # to gray scale
        img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = np.stack([img_gray, img_gray, img_gray], axis=2)

        # resize to (480, 640)
        resized_images, resize_infos = prutils.resize(image)
        resized_image = resized_images[0]
        scale, left, top = resize_infos[0]

        img_tensor = torch.tensor(resized_image).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)

        with torch.no_grad():
            x_dict = img_tensor_norm
            out_dict = self.network.forward_tracking(z_feat=self.z_feat, x=x_dict)

        pred_boxes = out_dict['pred_boxes'].tolist()[0]  # cx, cy, w, h (normalized 0-1)
        cx, cy, w, h = pred_boxes

        cx_pad, cy_pad = cx * 640, cy * 480
        w_pad, h_pad = w * 640, h * 480

        cx_orig = (cx_pad - left) / scale
        cy_orig = (cy_pad - top) / scale
        w_full = w_pad / scale
        h_full = h_pad / scale

        x = cx_orig - w_full / 2
        y = cy_orig - h_full / 2

        self.state = [x, y, w_full, h_full]

        if self.visualize_num > 0:
            save_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            save_image = cv.rectangle(save_image.copy(), (int(x), int(y)), (int(x + w_full), int(y + h_full)), (0, 255, 0), 2)
            if not os.path.exists(self.visualize_dir):
                os.makedirs(self.visualize_dir)

            cv.imwrite(os.path.join(self.visualize_dir, f'image_{self.visualize_num}.png'), save_image)
            self.visualize_num -= 1

        # response_origin = self.output_window * out_dict['score_map']

        # pred_box_origin = self.compute_box(response_origin, out_dict,
        #                                    resize_factor).tolist()  # .unsqueeze(dim=0)  # tolist()

        # self.state = pred_boxes

        return {"target_bbox": self.state}

    def compute_box(self, response, out_dict, resize_factor):
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = pred_boxes.mean(dim=0)
        pred_boxes[0] = pred_boxes[0] * 640 * resize_factor[1]  # x
        pred_boxes[1] = pred_boxes[1] * 480 * resize_factor[0]  # y
        pred_boxes[2] = pred_boxes[2] * 640 * resize_factor[1]  # w
        pred_boxes[3] = pred_boxes[3] * 480 * resize_factor[0]  # h
        return pred_boxes

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return lightFC
