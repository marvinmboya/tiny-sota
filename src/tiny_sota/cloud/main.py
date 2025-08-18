import torch 
import os 

from torchvision.models import resnet50, ResNet50_Weights


from ..utils.core import save_model
from ..vision import PretrainedModels

def createDir(dir):
    if not dir:
          dir = "models"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
    
def save_ResNet50(dir=None):
    dir = createDir(dir)
    path = f"{dir}/{PretrainedModels.ResNet_50.value}.pth"
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    save_model(model, path)
     
# def classModelsSave():
#     for model in PretrainedModels:
#          print(f"{model} -> {model.value}")
    # if os.path.exists("../tests/models/")