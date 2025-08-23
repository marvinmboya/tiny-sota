import torch 
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from dataclasses import dataclass
try:
    from collections.abc import Callable
except ImportError:
    from typing import Callable 

GRAY = torchvision.io.ImageReadMode.GRAY
RGB = torchvision.io.ImageReadMode.RGB

def read_image(path=None, gray=False)->torch.Tensor:
    mode = GRAY if gray else RGB 
    image = torchvision.io.read_image(path,mode)
    return image

def uint_to_float(in_: torch.Tensor)->torch.Tensor:
    out = in_.to(torch.float32)
    return out

def model_path(model_name: str, ext="pth")->str:
    path = f"models/{model_name}.{ext}"
    return path 

def save_model(model, path=None)->None:
    torch.save(model, path)

def save_weights(model, path=None)->None:
    torch.save(model.state_dict(), path)

def load_model(path=None)->torch.nn.Module:
    model = torch.load(path, weights_only=False)
    return model 

def load_weights(model, path)->torch.nn.Module:
    weights = torch.load(path, weights_only=True)
    model.load_state_dict(weights)
    return model

def get_device(device=None):
    if not device:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
    else:
        device = torch.device(device)
    return device 

def eval(model):
    return model.eval()

def get_model_device_state(model):
    next_state = next(model.parameters())
    model_device_state = "cpu" if next_state.is_cpu else (
        "cuda" if next_state.is_cuda else (
            "mps" if next_state.is_mps else (
                "xla" if next_state.is_xla else "None"
            )
        )
    )
    return model_device_state

def get_closest_graph_node(node_names, in_name):
    matching_children = [n for n in node_names if in_name.split(".")[0] in n]
    print(matching_children)

def get_node_names(model):
    node_names = get_graph_node_names(model)
    # only focused on eval node names
    eval_node_names = node_names[1] 
    return eval_node_names

def build_intermediate_model(model, in_name, out_name="out"):
    node_names = get_node_names(model)
    assert in_name in node_names, "CHECK GRAPH NODE NAME..."
    return_nodes = {in_name: out_name}
    inter_model = create_feature_extractor(model, return_nodes=return_nodes)
    def inference_inter_model(x):
        out_inter_model = inter_model(x)
        return out_inter_model[out_name]
    return inference_inter_model

@dataclass
class GraphMutation:
    getNodes: Callable = get_node_names
    getClosestNode: Callable = get_closest_graph_node
    buildIntermediateModel: Callable = build_intermediate_model