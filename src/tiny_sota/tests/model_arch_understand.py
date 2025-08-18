from torchvision.models import resnet50, ResNet50_Weights
from pprint import pprint 
import re


model = resnet50(weights=ResNet50_Weights.DEFAULT)

layers_obj = lambda ns: [eval(f"model{n}") for n in ns]
layers = lambda m, s: [n for n, _ in eval(f"m.{s}")]

n1 = layers(model, "named_modules()") 
n2 = layers(model, "named_children()") 


def parse(s_m):
    s = s_m.split(".")
    o = []
    for c in s:
        if c.isdigit():
            o.append(f"[{c}]")
        else:
            o.append(f".{c}")
    return ''.join(o)
        
def get_full_struct(model, parsed):
    result = []
    for p in parsed:
        # result.append(f"{p}======{eval(f"model[{p[1:]}]")}")
        result.append(f"\033[1;32mmodel{p}\033[0m======{eval(f'model{p}')}\n")
    return "".join(result)

parsed_layers = [parse(n) for n in n1]
# pprint(layers_obj(parsed_layers[1:]))
print(get_full_struct(model, parsed_layers[1:]))
print(model.layer4[2].relu)
