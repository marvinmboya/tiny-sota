import torch 
import torch.nn as nn 
from transformers import (
    AlbertConfig as AlbertConfig1, 
    AlbertModel as AlbertModel1
)
from tiny_sota.models.configs import AlbertConfig as AlbertConfig2
from tiny_sota.models.kokoro_load import transferKokoroWeights
from tiny_sota.models.kokoro_arch import Albert  as AlbertModel2

config = {"hidden_size": 768,"num_attention_heads": 12,
    "intermediate_size": 2048,"max_position_embeddings": 512,
    "num_hidden_layers": 12,"dropout": 0.1,"vocab_size": 178
}

config1 = AlbertConfig1(**config)
model = AlbertModel1(config1)

def func(model, input, output):
    pass
model.pooler_activation.register_forward_hook(func)

x = torch.randint(178, (1, 178), dtype=torch.long)
out1 = model(x).last_hidden_state

config2 = AlbertConfig2()
customalbert = AlbertModel2(config2)
transferKokoroWeights(customalbert, config, model)
out2, _ = customalbert(x)
print(out1.shape)
print(out2.shape)
print(torch.allclose(out1, out2, atol=1e-2))
# print(model)
# end compare the two

# def infer(model, x):
#     model.eval()
#     out = model(x)
#     return out.last_hidden_state

# y = infer(model, x)
