from enum import Enum 
import json 

class MetaClasses(Enum):
    ImageNet = "imagenet.txt"
    Coco = "coco.txt"
  



# curl https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json | jq ".[] | .[1]"

def parse_class_url(meta_class=MetaClasses.ImageNet, root="src/tiny_sota/meta"):
    meta_class = meta_class.value if isinstance(meta_class, Enum) else meta_class
    class_path = f"{root}/{meta_class}"
    with open(class_path, "r") as f:
        classes = json.load(f)
    return classes

def top_5_score(output,top_5=False):
    return output
