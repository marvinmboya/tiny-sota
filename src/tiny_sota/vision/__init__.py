from typing import Literal, Any 

import torch
# sys.path.append("..")

# change ..utils to utils
from ..tiny_utils.core import load_model, model_path #type: ignore
from ..tiny_utils.preprocess import pre_resnet #type: ignore
from ..tiny_utils.postprocess import top_5_score #type: ignore
from enum import Enum

def lazy_model_load(model_enum):
    def load():
        return load_model(model_path(model_enum.value))
    return load

class PretrainedModels(Enum):
    ResNet_50_v1_5: Literal["resnet-50-microsoft"] = "resnet-50-microsoft"
    ResNet_50: Literal["resnet-50-torch"] = "resnet-50-torch"
    Phi_3_5_Vision: Literal["Phi-3.5-vision-instruct"] = "Phi-3.5-vision-instruct"
    
MAPPING = {
    PretrainedModels.ResNet_50_v1_5: lazy_model_load(PretrainedModels.ResNet_50_v1_5),
    PretrainedModels.Phi_3_5_Vision: lazy_model_load(PretrainedModels.Phi_3_5_Vision)
}

def classification(
        image: torch.Tensor, 
        model_name: PretrainedModels
):
    image = pre_resnet(image)
    model = MAPPING[model_name]()
    output = model(image)
    output = top_5_score(output)
    return output 

    