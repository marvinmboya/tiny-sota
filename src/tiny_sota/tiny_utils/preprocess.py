import torch 
from torchvision.transforms.v2 import (
    Compose, Lambda, Resize, CenterCrop, Normalize
)

def freeze_layers(model, upto=None):
    def check_valid_upto(upto, no_):
        if upto >= no_:
            raise ValueError(
                f"\033[91m{upto}/{no_layers} fail to freeze, at least one should be unfrozen\033[0m")
    def freeze_number_layers(model, stop_i):
        count = 0
        for child in model.children():
            if not len(list(child.parameters())):
                continue
            if count < stop_i:
                count += 1
            else:
                break
            for param in child.parameters():
                param.requires_grad = False
        return model
    name_layers = [n for n, c in model.named_children() if len(list(c.parameters()))]
    no_layers = len(name_layers)
    if isinstance(upto, int):
        check_valid_upto(upto, no_layers)
        model = freeze_number_layers(model, upto)
    elif isinstance(upto, str):
        upto = name_layers.index(upto) + 1
        check_valid_upto(upto, no_layers)
        model = freeze_number_layers(model, upto)
    else:
        for param in model.parameters():
            param.requires_grad = False
    return model

def pre_resnet(image: torch.Tensor):
    # ResNet-50 v1.5 preprocessing
    compose = Compose([
        Resize(256),
        CenterCrop(224),
        Lambda(lambda x: x / 255.),
        Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
        )
    ])
    return compose(image)
    
