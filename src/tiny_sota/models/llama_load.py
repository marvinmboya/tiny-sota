import torch, torch.nn as nn 
from safetensors.torch import load_file
from .tiny_load import (
    assign, MODELS_META, getLocalWeightsDir, fetchLLMWeightAndTok
)
from ..tokenizers.llama import Llama3Tokenizer

def loadLlama3WeightsAndTok():
    Llama_Meta = MODELS_META.Llama32_1B
    local_dir = getLocalWeightsDir()
    loc_weight, loc_tok = fetchLLMWeightAndTok(Llama_Meta, local_dir)
    weight_dict = load_file(loc_weight)
    tokenizer = Llama3Tokenizer(loc_tok)
    return weight_dict, tokenizer

def transferLlama3Weights(model, param_config, params):
    model.embedding.weight = assign(model.embedding.weight, 
        params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    for l in range(param_config.layers):
        block = model.decoders[l]
        attn = block.attn
        attn.Wq.weight = assign(attn.Wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight")
        attn.Wk.weight = assign(attn.Wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight")
        attn.Wv.weight = assign(attn.Wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight")
        # Output projection
        attn.Wo.weight = assign(attn.Wo.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight")
        # Attention layernorm
        block.rms1.weight = assign(block.rms1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight")
        # Feedforward weights
        block.feed_forward.w1.weight = assign(
            block.feed_forward.w1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight")
        block.feed_forward.v.weight = assign(
            block.feed_forward.v.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight")
        block.feed_forward.w2.weight = assign(
            block.feed_forward.w2.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight")
        block.rms2.weight = assign(
            block.rms2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight")
    # Final normalization and output head
    model.rms_norm.weight = assign(model.rms_norm.weight, params["model.norm.weight"], "model.norm.weight")
    if "lm_head.weight" in params:
        model.linear.weight = assign(model.linear.weight, params["lm_head.weight"], "lm_head.weight")
    else: 
        model.linear.weight = assign(model.linear.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
