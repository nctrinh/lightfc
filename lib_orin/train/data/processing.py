import os

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import lib.train.data.transforms as tfm
import torch.nn.functional as F
import numpy as np

import cv2

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data

class TUProcessing(BaseProcessing):
    def __init__(self, search_area_factor, template_size, center_jitter_factor, scale_jitter_factor,
                     mode='pair', box=False, settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.template_size = template_size
        self.mode = mode
        self.settings = settings
        self.visualized = False
        self.visualize_num = 100
        self.visualize_list = []

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor)
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):

        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1

            if s == 'search':
                # Resize the search image to 480 x 640, and also resize the ground truth annotation accordingly
                data[s + '_images'], resize_info = prutils.resize(data[s + '_images'])

                new_annos = []
                for anno, (scale, pad_left, pad_top) in zip(data[s + '_anno'], resize_info):
                    new_annos.append(torch.tensor([
                        anno[0] * scale + pad_left,
                        anno[1] * scale + pad_top,
                        anno[2] * scale,
                        anno[3] * scale
                    ]))
                data[s + '_anno'] = new_annos

                crops = data[s + '_images']
                boxes = []
                for box in data[s + '_anno']:
                    x1 = max(0.0, float(box[0])) / 640.0
                    y1 = max(0.0, float(box[1])) / 480.0
                    x2 = min(640.0, float(box[0] + box[2])) / 640.0
                    y2 = min(480.0, float(box[1] + box[3])) / 480.0
                    boxes.append(torch.tensor([x1, y1, x2 - x1, y2 - y1]))
                boxes = torch.stack(boxes, dim=0)
                att_mask = [np.zeros((480, 640), dtype=np.bool_) for _ in crops]
                mask_crops = [torch.zeros((1, 480, 640), dtype=torch.float32) for _ in crops]

                min_area = 16 * 16
                box_w_px = boxes[:, 2] * 640.0
                box_h_px = boxes[:, 3] * 480.0

                if (box_w_px * box_h_px < min_area).any():
                    data['valid'] = False
                    return data

            elif s == 'template':
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor)

                if (crop_sz < 1).any():
                    data['valid'] = False
                    print("Too small box is found. Replace it with new data.")
                    return data     

                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
                    data[s + '_images'], jittered_anno, data[s + '_anno'],
                    self.search_area_factor, self.template_size, masks=data[s + '_masks']
                )

            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                                                            
            if s == 'template':
                for ele in data[s + '_att']:
                    if (ele == 1).all():
                        data['valid'] = False
                        print('Values of original attention mask are all one. Replace it with new data.')
                        return data

                    feat_size = self.template_size // 16
                    mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                    if (mask_down == 1).all():
                        data['valid'] = False
                        print('Values of down-sampled attention mask are all one. Replace it with new data.')
                        return data

        data['valid'] = True

        # Copy-and-paste augmentation fallback
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.template_size, self.template_size))
            data["search_masks"] = torch.zeros((1, 480, 640))
            
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        if not self.visualized:
            # Handle shape differences between 'sequence' (B, C, H, W) and 'pair' (C, H, W)
            t_imgs = data['template_images']
            s_imgs = data['search_images']
            t_annos = data['template_anno']
            s_annos = data['search_anno']

            for i in range(min(self.visualize_num, t_imgs.size(0))):
                # 1. Permute to HWC and convert to numpy
                template_img = t_imgs[i].permute(1, 2, 0).cpu().numpy()
                search_img = s_imgs[i].permute(1, 2, 0).cpu().numpy()

                # 2. Denormalize ImageNet float back to standard RGB
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                
                template_img = (template_img * std) + mean
                search_img = (search_img * std) + mean

                # Clip to [0.0, 1.0] to remove math noise, scale to 255, and cast to uint8
                template_img = (np.clip(template_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                search_img = (np.clip(search_img, 0.0, 1.0) * 255.0).astype(np.uint8)

                # 3. Convert RGB to BGR for OpenCV saving
                template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2BGR)
                search_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)

                template_box = t_annos[i].cpu().numpy().astype(np.float32)
                search_box = s_annos[i].cpu().numpy().astype(np.float32)

                # 4. Un-normalize bounding boxes if they are in [0.0, 1.0] format
                t_h, t_w = template_img.shape[:2]
                s_h, s_w = search_img.shape[:2]
                
                template_box *= np.array([t_w, t_h, t_w, t_h])
                search_box *= np.array([s_w, s_h, s_w, s_h])

                # 5. Draw bounding boxes (Format is x, y, w, h)
                template_img = cv2.rectangle(template_img.copy(), 
                                                (int(template_box[0]), int(template_box[1])),
                                                (int(template_box[0] + template_box[2]), int(template_box[1] + template_box[3])),
                                                (0, 255, 0), 2)
                search_img = cv2.rectangle(search_img.copy(), 
                                            (int(search_box[0]), int(search_box[1])),
                                            (int(search_box[0] + search_box[2]), int(search_box[1] + search_box[3])),
                                            (0, 255, 0), 2)

                self.visualize_list.append((template_img, search_img))

            # Save to dir
            vis_path = '/mnt/tupm/LightFC/visualization'
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

            for idx, (t_img, s_img) in enumerate(self.visualize_list):
                cv2.imwrite(os.path.join(vis_path, f'template_{self.visualize_num}.png'), t_img)
                cv2.imwrite(os.path.join(vis_path, f'search_{self.visualize_num}.png'), s_img)
                self.visualize_num -= 1

            self.visualize_list = []
            if self.visualize_num <= 0:
                self.visualized = True

        return data