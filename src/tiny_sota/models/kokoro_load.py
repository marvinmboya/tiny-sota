import torch
import torch.nn as nn 
from .tiny_load import MODELS_META, getLocalWeightsDir, fetchFilesHuggingFace
from .configs import KokoroConfig

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

def transferPredictorWeights(model, config: KokoroConfig, params):
    # TEXT ENCODER
    for i in [0, 2, 4]:
        model.predictor.text_encoder.lstms[i].weight_ih_l0 = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.weight_ih_l0'].clone()   
        )
        model.predictor.text_encoder.lstms[i].weight_hh_l0 = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.weight_hh_l0'].clone()   
        )
        model.predictor.text_encoder.lstms[i].bias_ih_l0 = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.bias_ih_l0'].clone()   
        )
        model.predictor.text_encoder.lstms[i].bias_hh_l0 = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.bias_hh_l0'].clone()   
        )
        model.predictor.text_encoder.lstms[i].weight_ih_l0_reverse = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.weight_ih_l0_reverse'].clone()   
        )
        model.predictor.text_encoder.lstms[i].weight_hh_l0_reverse = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.weight_hh_l0_reverse'].clone()   
        )
        model.predictor.text_encoder.lstms[i].bias_ih_l0_reverse = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.bias_ih_l0_reverse'].clone()   
        )
        model.predictor.text_encoder.lstms[i].bias_hh_l0_reverse = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.bias_hh_l0_reverse'].clone()   
        )
    for i in [1, 3, 5]:
        model.predictor.text_encoder.lstms[i].fc.weight = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.fc.weight'].clone()   
        )
        model.predictor.text_encoder.lstms[i].fc.bias = nn.Parameter(
            params[f'predictor'][f'module.text_encoder.lstms.{i}.fc.bias'].clone()   
        )
    # LSTM
    model.predictor.lstm.weight_ih_l0 = nn.Parameter(
        params['predictor'][f'module.lstm.weight_ih_l0'].clone()   
    )
    model.predictor.lstm.weight_hh_l0 = nn.Parameter(
        params['predictor']['module.lstm.weight_hh_l0'].clone()   
    )
    model.predictor.lstm.bias_ih_l0 = nn.Parameter(
        params['predictor']['module.lstm.bias_ih_l0'].clone()   
    )
    model.predictor.lstm.bias_hh_l0 = nn.Parameter(
        params['predictor']['module.lstm.bias_hh_l0'].clone()   
    )
    # LSTM REVERSE
    model.predictor.lstm.weight_ih_l0_reverse = nn.Parameter(
        params['predictor']['module.lstm.weight_ih_l0_reverse'].clone()   
    )
    model.predictor.lstm.weight_hh_l0_reverse = nn.Parameter(
        params['predictor']['module.lstm.weight_hh_l0_reverse'].clone()   
    )
    model.predictor.lstm.bias_ih_l0_reverse = nn.Parameter(
        params['predictor']['module.lstm.bias_ih_l0_reverse'].clone()   
    )
    model.predictor.lstm.bias_hh_l0_reverse = nn.Parameter(
        params['predictor']['module.lstm.bias_hh_l0_reverse'].clone()   
    )
    # DURATION PROJ
    model.predictor.duration_proj.weight = nn.Parameter(
        params['predictor']['module.duration_proj.linear_layer.weight'].clone()   
    )
    model.predictor.duration_proj.bias = nn.Parameter(
        params['predictor']['module.duration_proj.linear_layer.bias'].clone()   
    )
    # F0 AND N PROJ
    model.predictor.F0_proj.weight = nn.Parameter(
        params['predictor']['module.F0_proj.weight'].clone()   
    )
    model.predictor.F0_proj.bias = nn.Parameter(
        params['predictor']['module.F0_proj.bias'].clone()   
    )
    model.predictor.N_proj.weight = nn.Parameter(
        params['predictor']['module.N_proj.weight'].clone()   
    )
    model.predictor.N_proj.bias = nn.Parameter(
        params['predictor']['module.N_proj.bias'].clone()   
    )
    # F APPENDS
    for i in range(config.layers):
        model.predictor.F0[i].conv1.weight_g = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv1.weight_g'].clone()   
        )
        model.predictor.F0[i].conv1.weight_v = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv1.weight_v'].clone()   
        )
        model.predictor.F0[i].conv1.bias = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv1.bias'].clone()   
        )
        model.predictor.F0[i].conv2.weight_g = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv2.weight_g'].clone()   
        )
        model.predictor.F0[i].conv2.weight_v = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv2.weight_v'].clone()   
        )
        model.predictor.F0[i].conv2.bias = nn.Parameter(
            params['predictor'][f'module.F0.{i}.conv2.bias'].clone()   
        )
        model.predictor.F0[i].norm1.fc.weight = nn.Parameter(
            params['predictor'][f'module.F0.{i}.norm1.fc.weight'].clone()   
        )
        model.predictor.F0[i].norm1.fc.bias = nn.Parameter(
            params['predictor'][f'module.F0.{i}.norm1.fc.bias'].clone()   
        )
        model.predictor.F0[i].norm2.fc.weight = nn.Parameter(
            params['predictor'][f'module.F0.{i}.norm2.fc.weight'].clone()   
        )
        model.predictor.F0[i].norm2.fc.bias = nn.Parameter(
            params['predictor'][f'module.F0.{i}.norm2.fc.bias'].clone()   
        )
        # CONV1x1 AND POOL
        if i == 1:
            model.predictor.F0[i].conv1x1.weight_g = nn.Parameter(
                params['predictor'][f'module.F0.{i}.conv1x1.weight_g'].clone()   
            )
            model.predictor.F0[i].conv1x1.weight_v = nn.Parameter(
                params['predictor'][f'module.F0.{i}.conv1x1.weight_v'].clone()   
            )
            model.predictor.F0[i].pool.weight_g = nn.Parameter(
                params['predictor'][f'module.F0.{i}.pool.weight_g'].clone()   
            )
            model.predictor.F0[i].pool.weight_v = nn.Parameter(
                params['predictor'][f'module.F0.{i}.pool.weight_v'].clone()   
            )
            model.predictor.F0[i].pool.bias = nn.Parameter(
                params['predictor'][f'module.F0.{i}.pool.bias'].clone()   
            )
    # N APPENDS
    for i in range(config.layers):
        model.predictor.N[i].conv1.weight_g = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv1.weight_g'].clone()   
        )
        model.predictor.N[i].conv1.weight_v = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv1.weight_v'].clone()   
        )
        model.predictor.N[i].conv1.bias = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv1.bias'].clone()   
        )
        model.predictor.N[i].conv2.weight_g = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv2.weight_g'].clone()   
        )
        model.predictor.N[i].conv2.weight_v = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv2.weight_v'].clone()   
        )
        model.predictor.N[i].conv2.bias = nn.Parameter(
            params['predictor'][f'module.N.{i}.conv2.bias'].clone()   
        )
        model.predictor.N[i].norm1.fc.weight = nn.Parameter(
            params['predictor'][f'module.N.{i}.norm1.fc.weight'].clone()   
        )
        model.predictor.N[i].norm1.fc.bias = nn.Parameter(
            params['predictor'][f'module.N.{i}.norm1.fc.bias'].clone()   
        )
        model.predictor.N[i].norm2.fc.weight = nn.Parameter(
            params['predictor'][f'module.N.{i}.norm2.fc.weight'].clone()   
        )
        model.predictor.N[i].norm2.fc.bias = nn.Parameter(
            params['predictor'][f'module.N.{i}.norm2.fc.bias'].clone()   
        )
        # CONV1x1 AND POOL
        if i == 1:
            model.predictor.N[i].conv1x1.weight_g = nn.Parameter(
                params['predictor'][f'module.N.{i}.conv1x1.weight_g'].clone()   
            )
            model.predictor.N[i].conv1x1.weight_v = nn.Parameter(
                params['predictor'][f'module.N.{i}.conv1x1.weight_v'].clone()   
            )
            model.predictor.N[i].pool.weight_g = nn.Parameter(
                params['predictor'][f'module.N.{i}.pool.weight_g'].clone()   
            )
            model.predictor.N[i].pool.weight_v = nn.Parameter(
                params['predictor'][f'module.N.{i}.pool.weight_v'].clone()   
            )
            model.predictor.N[i].pool.bias = nn.Parameter(
                params['predictor'][f'module.N.{i}.pool.bias'].clone()   
            )
    # SHARED
    model.predictor.shared.weight_ih_l0 = nn.Parameter(
        params['predictor'][f'module.shared.weight_ih_l0'].clone()   
    )
    model.predictor.shared.weight_hh_l0 = nn.Parameter(
        params['predictor']['module.shared.weight_hh_l0'].clone()   
    )
    model.predictor.shared.bias_ih_l0 = nn.Parameter(
        params['predictor']['module.shared.bias_ih_l0'].clone()   
    )
    model.predictor.shared.bias_hh_l0 = nn.Parameter(
        params['predictor']['module.shared.bias_hh_l0'].clone()   
    )
    # SHARED REVERSE
    model.predictor.shared.weight_ih_l0_reverse = nn.Parameter(
        params['predictor']['module.shared.weight_ih_l0_reverse'].clone()   
    )
    model.predictor.shared.weight_hh_l0_reverse = nn.Parameter(
        params['predictor']['module.shared.weight_hh_l0_reverse'].clone()   
    )
    model.predictor.shared.bias_ih_l0_reverse = nn.Parameter(
        params['predictor']['module.shared.bias_ih_l0_reverse'].clone()   
    )
    model.predictor.shared.bias_hh_l0_reverse = nn.Parameter(
        params['predictor']['module.shared.bias_hh_l0_reverse'].clone()   
    )
            




def transferKokoroWeights(model, config: KokoroConfig, params):
    transferAlbertWeights(model, config, params)
    model.ff.weight = nn.Parameter(
        params['bert_encoder']['module.weight'].clone()
    )
    model.ff.bias = nn.Parameter(
        params['bert_encoder']['module.bias'].clone()
    )
    # TEXT ENCODER
    transferPredictorWeights(model, config, params)
    # odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'])
        

import torch

def check_F0_weights(model, params, config, tol=1e-6):
    all_match = True
    for i in range(config.layers):
        layer = model.predictor.F0[i]

        checks = [
            ("conv1.weight_g", layer.conv1.weight_g, params['predictor'][f'module.F0.{i}.conv1.weight_g']),
            ("conv1.weight_v", layer.conv1.weight_v, params['predictor'][f'module.F0.{i}.conv1.weight_v']),
            ("conv1.bias", layer.conv1.bias, params['predictor'][f'module.F0.{i}.conv1.bias']),
            ("conv2.weight_g", layer.conv2.weight_g, params['predictor'][f'module.F0.{i}.conv2.weight_g']),
            ("conv2.weight_v", layer.conv2.weight_v, params['predictor'][f'module.F0.{i}.conv2.weight_v']),
            ("conv2.bias", layer.conv2.bias, params['predictor'][f'module.F0.{i}.conv2.bias']),
            ("norm1.fc.weight", layer.norm1.fc.weight, params['predictor'][f'module.F0.{i}.norm1.fc.weight']),
            ("norm1.fc.bias", layer.norm1.fc.bias, params['predictor'][f'module.F0.{i}.norm1.fc.bias']),
            ("norm2.fc.weight", layer.norm2.fc.weight, params['predictor'][f'module.F0.{i}.norm2.fc.weight']),
            ("norm2.fc.bias", layer.norm2.fc.bias, params['predictor'][f'module.F0.{i}.norm2.fc.bias']),
        ]

        # conv1x1 and pool only exist in layer 1
        if i == 1:
            checks += [
                ("conv1x1.weight_g", layer.conv1x1.weight_g, params['predictor'][f'module.F0.{i}.conv1x1.weight_g']),
                ("conv1x1.weight_v", layer.conv1x1.weight_v, params['predictor'][f'module.F0.{i}.conv1x1.weight_v']),
                ("pool.weight_g", layer.pool.weight_g, params['predictor'][f'module.F0.{i}.pool.weight_g']),
                ("pool.weight_v", layer.pool.weight_v, params['predictor'][f'module.F0.{i}.pool.weight_v']),
                ("pool.bias", layer.pool.bias, params['predictor'][f'module.F0.{i}.pool.bias']),
            ]

        for name, tensor, source in checks:
            if not torch.allclose(tensor, source, atol=tol):
                print(f"[Layer {i}] {name} mismatch!")
                all_match = False

    if all_match:
        print("✅ All F0 weights match the source params.")
    else:
        print("❌ Some F0 weights do not match.")
