# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import cv2
import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection

from damo.structures.bounding_box import BoxList

try:
    import pycocotools.mask as mask_util
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False

cv2.setNumThreads(0)


def rle_to_polygon(rle, img_shape):
    """
    Convert RLE mask to polygon format using pycocotools.
    
    Args:
        rle: RLE format mask (dict with 'counts' and 'size')
        img_shape: tuple of (height, width)
    
    Returns:
        list: Polygon coordinates as [x1, y1, x2, y2, ...]
    """
    if not PYCOCOTOOLS_AVAILABLE:
        return []
    
    try:
        # Decode RLE to binary mask
        if isinstance(rle, dict):
            mask = mask_util.decode(rle)
        else:
            return []
            
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert largest contour to polygon
        if len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the contour to reduce points
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Flatten to [x1, y1, x2, y2, ...] format
            polygon = approx.flatten().tolist()
            
            # Ensure we have at least 6 points (3 vertices for a triangle)
            if len(polygon) >= 6:
                return polygon
                
    except Exception as e:
        # If conversion fails, return empty list
        pass
    
    return []


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, transforms=None, class_names=None):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        assert (class_names is not None), 'plz provide class_names'

        self.contiguous_class2id = {
            class_name: i
            for i, class_name in enumerate(class_names)
        }
        self.contiguous_id2class = {
            i: class_name
            for i, class_name in enumerate(class_names)
        }

        categories = self.coco.dataset['categories']
        cat_names = [cat['name'] for cat in categories]
        cat_ids = [cat['id'] for cat in categories]
        self.ori_class2id = {
            class_name: i
            for class_name, i in zip(cat_names, cat_ids)
        }
        self.ori_id2class = {
            i: class_name
            for class_name, i in zip(cat_names, cat_ids)
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        classes = torch.tensor(classes)
        target.add_field('labels', classes)


        target = target.clip_to_image(remove_empty=True)

        # PIL to numpy array
        img = np.asarray(img)  # rgb

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def pull_item(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            if 'segmentation' in obj:
                seg = obj['segmentation']
                
                # Handle different segmentation formats
                if isinstance(seg, dict):
                    # RLE format (compressed or uncompressed)
                    if PYCOCOTOOLS_AVAILABLE:
                        try:
                            # Convert RLE to polygon
                            img_info = self.get_img_info(idx)
                            img_shape = (img_info['height'], img_info['width'])
                            polygon = rle_to_polygon(seg, img_shape)
                            if len(polygon) > 0:
                                obj_mask_array = np.array(polygon, dtype=np.float32)
                                if len(obj_mask_array) % 2 == 0:
                                    obj_masks.append(obj_mask_array)
                        except Exception:
                            pass
                    continue
                    
                elif isinstance(seg, list):
                    # Check if it's a list of polygons or RLE
                    if len(seg) > 0:
                        # Check if this is an RLE in list format
                        if len(seg) == 1 and isinstance(seg[0], dict):
                            # Single RLE object
                            if PYCOCOTOOLS_AVAILABLE:
                                try:
                                    # Convert RLE to polygon
                                    img_info = self.get_img_info(idx)
                                    img_shape = (img_info['height'], img_info['width'])
                                    polygon = rle_to_polygon(seg[0], img_shape)
                                    if len(polygon) > 0:
                                        obj_mask_array = np.array(polygon, dtype=np.float32)
                                        if len(obj_mask_array) % 2 == 0:
                                            obj_masks.append(obj_mask_array)
                                except Exception:
                                    pass
                            continue
                            
                        elif isinstance(seg[0], list):
                            # Polygon format - list of [x1,y1,x2,y2,...]
                            obj_mask = []
                            for mask in seg:
                                if isinstance(mask, list) and all(isinstance(x, (int, float)) for x in mask):
                                    obj_mask.extend(mask)
                            
                            if len(obj_mask) > 0:
                                try:
                                    # Ensure all values can be converted to float
                                    obj_mask_array = np.array(obj_mask, dtype=np.float32)
                                    if len(obj_mask_array) % 2 == 0:  # Must have even number for x,y pairs
                                        obj_masks.append(obj_mask_array)
                                except (ValueError, TypeError):
                                    # Skip if conversion fails
                                    continue
                                    
                        elif isinstance(seg[0], (int, float)) and all(isinstance(x, (int, float)) for x in seg):
                            # Single polygon as flat list
                            try:
                                obj_mask_array = np.array(seg, dtype=np.float32)
                                if len(obj_mask_array) % 2 == 0:
                                    obj_masks.append(obj_mask_array)
                            except (ValueError, TypeError):
                                continue
                        else:
                            # Could be RLE in string format or other format
                            if PYCOCOTOOLS_AVAILABLE and isinstance(seg[0], str):
                                try:
                                    # Try to decode as RLE - skip for now
                                    pass
                                except Exception:
                                    pass
                            continue
        
        seg_masks = [
            obj_mask.reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, idx

    def load_anno(self, idx):
        _, anno = super(COCODataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        return classes

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
