from datetime import datetime
from pathlib import Path
import os
from tqdm.notebook import tqdm

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 

import roboflow
from roboflow import Roboflow
import supervision as sv

def imgaug_bbx2yolo(bbs_aug_adj, sh_w, sh_h):
    w = (bbs_aug_adj.x2 - bbs_aug_adj.x1)/sh_w
    h = (bbs_aug_adj.y2 - bbs_aug_adj.y1)/sh_h
    x_mid = (bbs_aug_adj.x2 + bbs_aug_adj.x1)/(2*sh_w)
    y_mid = (bbs_aug_adj.y2 + bbs_aug_adj.y1)/(2*sh_h)

    return w, h, x_mid, y_mid

class aug_gen():
    def __init__(self, DATASET_NAME):
        self.IMG_DIR_PATH = f"autolabel_dataset\{DATASET_NAME}\images"
        self.ANN_DIR_PATH = f"autolabel_dataset\{DATASET_NAME}\images"
        self.DATA_YAML_PATH = f"autolabel_dataset\{DATASET_NAME}\data.yaml"
        self.OUTPUT_GEN_PATH = f"autolabel_dataset\{DATASET_NAME}\\aug"
        Path(self.OUTPUT_GEN_PATH).mkdir(parents=True, exist_ok=True)

        self.ds = sv.DetectionDataset.from_yolo(
            images_directory_path=self.IMG_DIR_PATH,
            annotations_directory_path=self.ANN_DIR_PATH,
            data_yaml_path=self.DATA_YAML_PATH
        )

    def gen_aug_img_dataset(self, seq: iaa.Sequential, seq_effect: iaa.Sequential, AUG_IMG_EFFECT: int, repeat_num=1):
        progress_bar = tqdm(self.ds, desc="Gen aug images")
        for detect in progress_bar:
            progress_bar.set_description(desc=f"Labeling {detect[0]}", refresh=True)
            x = detect
            # Create BoundingBoxes list
            list_bbox = []
            for idx_box, idx_cls in zip( x[2].xyxy, x[2].class_id):
                list_bbox.append(BoundingBox(x1=idx_box[0], x2=idx_box[2], y1=idx_box[1], y2=idx_box[3], label=idx_cls))
            
            image = imageio.v2.imread(x[0])
        
            # get the ori BoundingBoxes
            bbs = BoundingBoxesOnImage(list_bbox, shape=image.shape)
        
            for idx in range(repeat_num):
                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
                if AUG_IMG_EFFECT:
                    image_aug, bbs_aug = seq_effect(image=image_aug, bounding_boxes=bbs_aug)
                
                # remove out of image bbox and fix the outside val to 0
                bbs_aug_adj = bbs_aug.remove_out_of_image_fraction(0.8).clip_out_of_image()
                
                # Convert to yolo format annotation
                yolo_anno_str_list = []
                for bbs_aug_adj_s in bbs_aug_adj:
                    w, h, x_mid, y_mid = imgaug_bbx2yolo(bbs_aug_adj_s, bbs_aug_adj.shape[1], bbs_aug_adj.shape[0])
                    label = bbs_aug_adj_s.label
                    yolo_str = "{} {} {} {} {} \n".format(label, x_mid, y_mid, w, h)
                    yolo_anno_str_list.append(yolo_str)
                
                # save the img file
                try:
                    now = datetime.now()
                    timestamp = datetime.timestamp(now)
                    timestamp = str(int(timestamp))
                    img_name = "{}_{}_{}.jpg".format(Path(x[0]).stem, timestamp, str(idx))
                    img_path = os.path.join(self.OUTPUT_GEN_PATH, img_name)
                    imageio.imwrite(img_path, image_aug)
                except Exception as e:
                    print(e)    
                # save the annotation
                try:
                    ann_name = "{}_{}_{}.txt".format(Path(x[0]).stem, timestamp, str(idx))
                    ann_path = os.path.join(self.OUTPUT_GEN_PATH, ann_name)
                    fo = open(ann_path, "w")
                    fo.writelines(yolo_anno_str_list)
                    fo.close()
                except Exception as e:
                    print(e)

        print("Finish!! The new dataset is here: {}".format(self.OUTPUT_GEN_PATH))            