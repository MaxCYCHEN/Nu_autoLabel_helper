from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.helpers import split_data
import supervision as sv

from pathlib import Path
import os
import cv2
import glob
from tqdm.notebook import tqdm

class GroundingDINO_label():

    def __init__(self, IMG_DIR_PATH, OUT_DIR_NAME, Class_Ontology):
        self.IMG_DIR_PATH = IMG_DIR_PATH
        self.OUT_DIR_NAME = OUT_DIR_NAME
        self.Class_Ontology = Class_Ontology
        self.CLASS = list(self.Class_Ontology.values())

    def auto_labelgen_dataset(self, box_threshold, text_threshold, extension) -> sv.DetectionDataset:
    
        images_map = {}
        detections_map = {}
        
        base_model = GroundingDINO(ontology=CaptionOntology(self.Class_Ontology), box_threshold=box_threshold, text_threshold=text_threshold)
        
        files = glob.glob(self.IMG_DIR_PATH + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")
        
        for image_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {image_path}", refresh=True)
            
            image = cv2.imread(image_path)
            h, w, _ = image.shape
        
            f_path_short = os.path.basename(image_path)
            images_map[f_path_short] = image.copy()
            
            detections = base_model.predict(image_path)
        
            # drop potential detections with phrase that is not part of CLASSES set
            detections = detections[detections.class_id != None]
            # drop potential detections with area close to area of whole image
            detections = detections[(detections.area / (h * w)) < 0.9 ]
            # drop potential double detections
            detections = detections.with_nms(class_agnostic=True)
        
            detections_map[f_path_short] = detections
        
        # Create dataset instance
        return sv.DetectionDataset(self.CLASS, images_map, detections_map)


def create_dataset(OUT_DIR_NAME: str, Class_Ontology: dict, dataset: sv.DetectionDataset, 
                   data_style: str = 'yolo', split_en: int = 0, split_ratio: float = 0.8):
    
    CLASS = list(Class_Ontology.values())

    if data_style.lower().count('yolo'):
        OUTPUT_DIR_PATH = os.path.join("autolabel_dataset", OUT_DIR_NAME)
        try:
            dataset.as_yolo(
                        os.path.join(OUTPUT_DIR_PATH, "images"),
                        os.path.join(OUTPUT_DIR_PATH, "images"), # annotations put in same folder for yolo format
                        #os.path.join(OUTPUT_DIR_PATH, "annotations"),
                        min_image_area_percentage=0.01,
                        data_yaml_path=os.path.join(OUTPUT_DIR_PATH, "data.yaml")
            )
        except Exception as e:
            print(e)

        # create a classes.txt for using labelimg
        fo = open(os.path.join(OUTPUT_DIR_PATH, "images", "classes.txt"), "w")
        fo.writelines([ line + "\n" for line in CLASS ])
        fo.close()
        print("Finish! The auto lable dataset: {}".format(OUTPUT_DIR_PATH))
    elif data_style.lower().count('coco'):
        OUTPUT_DIR_PATH = os.path.join("autolabel_dataset", (OUT_DIR_NAME + r"_coco"))
        try:
            dataset.as_coco(
                        os.path.join(OUTPUT_DIR_PATH, "images"),
                        os.path.join(OUTPUT_DIR_PATH, "annotations", "annotations.json"),
                        min_image_area_percentage=0
            )
        except Exception as e:
            print(e)
        print("Finish! The auto lable dataset: {}".format(OUTPUT_DIR_PATH))
    else:
        print("Only support yolo and coco format!!! Now using: {}".format(data_style))        

    if split_en:
        print("Spliting on going!")
        try:
            split_data(OUTPUT_DIR_PATH, split_ratio=split_ratio)
            print("Finish! The auto lable dataset: {}".format(OUTPUT_DIR_PATH))     
        except Exception as e:
            print(e)    

def yolo2coco_dataset(IMG_DIR_PATH, ANN_DIR_PATH, OUTPUT_DIR_PATH_COCO):
    
    try:
        ds = sv.DetectionDataset.from_yolo(
        images_directory_path=IMG_DIR_PATH,
        annotations_directory_path=ANN_DIR_PATH,
        data_yaml_path=os.path.join(Path(IMG_DIR_PATH).parents[0], "data.yaml")
        )
        print("Finish load as Yolo format!")
    except Exception as e:
        print(e)

    try:
        ds.as_coco(
        os.path.join(OUTPUT_DIR_PATH_COCO, "train2017"),
        os.path.join(OUTPUT_DIR_PATH_COCO, "annotations", "annotations.json"),
        min_image_area_percentage=0
        )
        print("Finish convert to Coco format!")
    except Exception as e:
        print(e)          