import torch, torch.nn as nn 
from safetensors.torch import load_file
from .tiny_load import (
    LLMS_META, getLocalWeightsDir, fetchLLMWeightAndTok
)

def fetchQwen3WeightsAndTok():
    Qwen_Meta = LLMS_META.Qwen3_06B
    local_dir = getLocalWeightsDir()
    fetchLLMWeightAndTok(Qwen_Meta, local_dir)

def loadQwen3WeightsAndTok():
    Qwen_Meta = LLMS_META.Qwen3_06B
    local_dir = getLocalWeightsDir()
    loc_weight  = local_dir/Qwen_Meta["loc_weight"]
    loc_tok  = local_dir/Qwen_Meta["loc_tok"]
    assert loc_weight.exists(), "Qwen weights not downloaded!"
    assert loc_tok.exists(), "Qwen tokenizer not downloaded!"
    weight_dict = load_file(loc_weight)
    return weight_dict, loc_tok

def transferQwen3Weights(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))
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
        # QK norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign(attn.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight")
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign(attn.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight")
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
        print("Model uses weight tying.")
        model.linear.weight = assign(model.linear.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")


def transferTempQwenWeights(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))
    model.embed_tokens.weight = assign(model.embed_tokens.weight, 
        params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    for l in range(param_config.layers):
        block = model.layers[l]
        attn = block.self_attn
        attn.q_proj.weight = assign(attn.q_proj.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight")
        attn.k_proj.weight = assign(attn.k_proj.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight")
        attn.v_proj.weight = assign(attn.v_proj.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight")
        # Output projection
        attn.o_proj.weight = assign(attn.o_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight")
        # QK norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign(attn.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight")
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign(attn.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight")
        # Attention layernorm
        block.input_layernorm.weight = assign(block.input_layernorm.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight")
        # Feedforward weights
        block.mlp.gate_proj.weight = assign(
            block.mlp.gate_proj.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight")
        block.mlp.up_proj.weight = assign(
            block.mlp.up_proj.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight")
        block.mlp.down_proj.weight = assign(
            block.mlp.down_proj.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight")
        block.post_attention_layernorm.weight = assign(
            block.post_attention_layernorm.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight")
    # Final normalization and output head
    model.lm_head.weight = assign(model.lm_head.weight, params["model.norm.weight"], "model.norm.weight")
    if "lm_head.weight" in params:
        model.lm_head.weight = assign(model.lm_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        print("Model uses weight tying.")
        model.linear.weight = assign(model.linear.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
