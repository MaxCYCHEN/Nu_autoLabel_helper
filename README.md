# Nu_autoLabel_helper
Help user automatically labeling objects for object detection task

### Object Detection Labeling & Easy Image Augmentation
- Ref: [autodistill](https://github.com/autodistill/autodistill), [autodistill-grounding-dino](https://github.com/autodistill/autodistill-grounding-dino)
#### Installation:
- `python -m pip install -r requirements.txt`
- `python -m pip install git+https://github.com/MaxCYCHEN/autodistill-grounding-dino.git`
- `python -m pip install git+https://github.com/MaxCYCHEN/imgaug.git`
#### Usage
- Please open `custom_autoLabel_gdino.ipynb` and follow the steps and update the parameters basing on your dataset
- CLASS_ONTOLOGY example: `CLASS_ONTOLOGY = {<The prompt or description of object>:<Label>, ...}`
- After auto labeling finish, user can increase the volume of dataset by [imgaug](https://github.com/aleju/imgaug) help.
- Choose to use yolo or coco format.

