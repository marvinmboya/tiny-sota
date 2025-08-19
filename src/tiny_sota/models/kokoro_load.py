import torch
import torch.nn as nn 
from .tiny_load import MODELS_META, getLocalWeightsDir, fetchFilesHuggingFace

def fetchKokoroWeights():
    meta = MODELS_META.Kokoro_82M
    local_dir = getLocalWeightsDir()
    loc_weights = fetchFilesHuggingFace(
        repo_id = meta["repo_id"],
        commit = meta["commit"],
        rem_id = meta["weight_id"],
        loc_id = meta["loc_weight"],
        local_dir=local_dir
    )
    return loc_weights

def loadKokoroWeightsAndTok():
    loc_weight  = fetchKokoroWeights()
    weight_dict = torch.load(loc_weight, map_location="cpu", weights_only=True)
    return weight_dict

def transferAlbertWeights(model, config, params):
    bert = params['bert']
    model.albert.embedding.weight = nn.Parameter(
        bert['module.embeddings.word_embeddings.weight'].clone()
    )
    model.albert.pos_emb.weight = nn.Parameter(
        bert['module.embeddings.position_embeddings.weight'].clone()
    )
    model.albert.tok_emb.weight = nn.Parameter(
        bert['module.embeddings.token_type_embeddings.weight'].clone()
    )
    # LayerNorm 
    model.albert.ln.weight = nn.Parameter(
        bert['module.embeddings.LayerNorm.weight'].clone()
    )
    model.albert.ln.bias = nn.Parameter(
        bert['module.embeddings.LayerNorm.bias'].clone()
    )
    # ENCODER
    model.albert.encoder.ff_in.weight = nn.Parameter(
        bert['module.encoder.embedding_hidden_mapping_in.weight'].clone()
    )
    model.albert.encoder.ff_in.bias = nn.Parameter(
        bert['module.encoder.embedding_hidden_mapping_in.bias'].clone()
    )

    model.albert.encoder.attn.Wq.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight'].clone()
    )
    model.albert.encoder.attn.Wq.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias'].clone()
    )
    model.albert.encoder.attn.Wk.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight'].clone()
    )
    model.albert.encoder.attn.Wk.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias'].clone()
    )
    model.albert.encoder.attn.Wv.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight'].clone()
    )
    model.albert.encoder.attn.Wv.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias'].clone()
    )
    model.albert.encoder.attn.Wo.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight'].clone()
    )
    model.albert.encoder.attn.Wo.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias'].clone()
    )
    model.albert.encoder.mlp[0].weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight'].clone()
    )
    model.albert.encoder.mlp[0].bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias'].clone()
    )
    model.albert.encoder.mlp[2].weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight'].clone()
    )
    model.albert.encoder.mlp[2].bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias'].clone()
    )
    # END ENCODER
    # LayerNorm 
    model.albert.encoder.attn_ln.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight'].clone()
    )
    model.albert.encoder.attn_ln.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias'].clone()
    )
    model.albert.encoder.mlp_ln.weight = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight'].clone()
    )
    model.albert.encoder.mlp_ln.bias = nn.Parameter(
        bert['module.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias'].clone()
    )
    # end LayerNorm

    model.albert.ff_out.weight = nn.Parameter(
        bert['module.pooler.weight'].clone()
    )
    model.albert.ff_out.bias = nn.Parameter(
        bert['module.pooler.bias'].clone()
    )

def transferKokoroWeights(model, config, params):
    transferAlbertWeights(model, config, params)