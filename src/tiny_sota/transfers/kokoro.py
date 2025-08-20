import torch 
import torch.nn as nn 
from ..models.configs import AlbertConfig, KokoroConfig

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
            
def transferTextEncoderWeights(model, config: KokoroConfig, params):
    model.text_encoder.embedding.weight = nn.Parameter(
        params['text_encoder']['module.embedding.weight'].clone()
    )
    for i in range(config.layers):
        model.text_encoder.cnn[i][0].weight_g = nn.Parameter(
           params['text_encoder'][f'module.cnn.{i}.0.weight_g'].clone() 
        )
        model.text_encoder.cnn[i][0].weight_v = nn.Parameter(
           params['text_encoder'][f'module.cnn.{i}.0.weight_v'].clone() 
        )
        model.text_encoder.cnn[i][0].bias = nn.Parameter(
           params['text_encoder'][f'module.cnn.{i}.0.bias'].clone() 
        )
        model.text_encoder.cnn[i][1].gamma = nn.Parameter(
           params['text_encoder'][f'module.cnn.{i}.1.gamma'].clone() 
        )
        model.text_encoder.cnn[i][1].beta = nn.Parameter(
           params['text_encoder'][f'module.cnn.{i}.1.beta'].clone() 
        )
    # LSTM
    model.text_encoder.lstm.weight_ih_l0 = nn.Parameter(
        params['text_encoder']['module.lstm.weight_ih_l0'].clone()   
    )
    model.text_encoder.lstm.weight_hh_l0 = nn.Parameter(
        params['text_encoder']['module.lstm.weight_hh_l0'].clone()   
    )
    model.text_encoder.lstm.bias_ih_l0 = nn.Parameter(
        params['text_encoder']['module.lstm.bias_ih_l0'].clone()   
    )
    model.text_encoder.lstm.bias_hh_l0 = nn.Parameter(
        params['text_encoder']['module.lstm.bias_hh_l0'].clone()   
    )
    # LSTM REVERSE
    model.text_encoder.lstm.weight_ih_l0_reverse = nn.Parameter(
        params['text_encoder']['module.lstm.weight_ih_l0_reverse'].clone()   
    )
    model.text_encoder.lstm.weight_hh_l0_reverse = nn.Parameter(
        params['text_encoder']['module.lstm.weight_hh_l0_reverse'].clone()   
    )
    model.text_encoder.lstm.bias_ih_l0_reverse = nn.Parameter(
        params['text_encoder']['module.lstm.bias_ih_l0_reverse'].clone()   
    )
    model.text_encoder.lstm.bias_hh_l0_reverse = nn.Parameter(
        params['text_encoder']['module.lstm.bias_hh_l0_reverse'].clone()   
    )


def transferDecoderWeights(decoder, config, params):
    for i in range(4):
        decoder.decode[i].conv1.weight_g = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv1.weight_g'].clone()   
        )
        decoder.decode[i].conv1.weight_v = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv1.weight_v'].clone()   
        )
        decoder.decode[i].conv1.bias = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv1.bias'].clone()   
        )
        decoder.decode[i].conv2.weight_g = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv2.weight_g'].clone()   
        )
        decoder.decode[i].conv2.weight_v = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv2.weight_v'].clone()   
        )
        decoder.decode[i].conv2.bias = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv2.bias'].clone()   
        )
        decoder.decode[i].norm1.fc.weight = nn.Parameter(
            params['decoder'][f'module.decode.{i}.norm1.fc.weight'].clone()   
        )
        decoder.decode[i].norm1.fc.bias = nn.Parameter(
            params['decoder'][f'module.decode.{i}.norm1.fc.bias'].clone()   
        )
        decoder.decode[i].norm2.fc.weight = nn.Parameter(
            params['decoder'][f'module.decode.{i}.norm2.fc.weight'].clone()   
        )
        decoder.decode[i].norm2.fc.bias = nn.Parameter(
            params['decoder'][f'module.decode.{i}.norm2.fc.bias'].clone()   
        )
        decoder.decode[i].conv1x1.weight_g = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv1x1.weight_g'].clone()   
        )
        decoder.decode[i].conv1x1.weight_v = nn.Parameter(
            params['decoder'][f'module.decode.{i}.conv1x1.weight_v'].clone()   
        )
    decoder.decode[3].pool.weight_g = nn.Parameter(
        params['decoder']['module.decode.3.pool.weight_g'].clone()   
    )
    decoder.decode[3].pool.weight_v = nn.Parameter(
        params['decoder']['module.decode.3.pool.weight_v'].clone()   
    )
    decoder.decode[i].pool.bias = nn.Parameter(
        params['decoder']['module.decode.3.pool.bias'].clone()   
    )
    ############
    decoder.encode.conv1.weight_g = nn.Parameter(
        params['decoder']['module.encode.conv1.weight_g'].clone()   
    )
    decoder.encode.conv1.weight_v = nn.Parameter(
        params['decoder']['module.encode.conv1.weight_v'].clone()   
    )
    decoder.encode.conv1.bias = nn.Parameter(
        params['decoder']['module.encode.conv1.bias'].clone()   
    )
    decoder.encode.conv2.weight_g = nn.Parameter(
        params['decoder']['module.encode.conv2.weight_g'].clone()   
    )
    decoder.encode.conv2.weight_v = nn.Parameter(
        params['decoder']['module.encode.conv2.weight_v'].clone()   
    )
    decoder.encode.conv2.bias = nn.Parameter(
        params['decoder']['module.encode.conv2.bias'].clone()   
    )
    decoder.encode.norm1.fc.weight = nn.Parameter(
        params['decoder']['module.encode.norm1.fc.weight'].clone()   
    )
    decoder.encode.norm1.fc.bias = nn.Parameter(
        params['decoder']['module.encode.norm1.fc.bias'].clone()   
    )
    decoder.encode.norm2.fc.weight = nn.Parameter(
        params['decoder']['module.encode.norm2.fc.weight'].clone()   
    )
    decoder.encode.norm2.fc.bias = nn.Parameter(
        params['decoder']['module.encode.norm2.fc.bias'].clone()   
    )
    decoder.encode.conv1x1.weight_g = nn.Parameter(
        params['decoder']['module.encode.conv1x1.weight_g'].clone()   
    )
    decoder.encode.conv1x1.weight_v = nn.Parameter(
        params['decoder']['module.encode.conv1x1.weight_v'].clone()   
    )
    ##################
    decoder.F0_conv.weight_g = nn.Parameter(params['decoder']['module.F0_conv.weight_g'].clone())
    decoder.F0_conv.weight_v = nn.Parameter(params['decoder']['module.F0_conv.weight_v'].clone())
    decoder.F0_conv.bias = nn.Parameter(params['decoder']['module.F0_conv.bias'].clone())
    # Copy N_conv parameters
    decoder.N_conv.weight_g = nn.Parameter(params['decoder']['module.N_conv.weight_g'].clone())
    decoder.N_conv.weight_v = nn.Parameter(params['decoder']['module.N_conv.weight_v'].clone())
    decoder.N_conv.bias = nn.Parameter(params['decoder']['module.N_conv.bias'].clone())
    # Copy asr_res parameters
    decoder.asr_res[0].weight_g = nn.Parameter(params['decoder']['module.asr_res.0.weight_g'].clone())
    decoder.asr_res[0].weight_v = nn.Parameter(params['decoder']['module.asr_res.0.weight_v'].clone())
    decoder.asr_res[0].bias = nn.Parameter(params['decoder']['module.asr_res.0.bias'].clone())

def transferGeneratorWeights(generator, config, params):
    generator.m_source.l_linear.weight = nn.Parameter(params['decoder']['module.generator.m_source.l_linear.weight'].clone())
    generator.m_source.l_linear.bias = nn.Parameter(params['decoder']['module.generator.m_source.l_linear.bias'].clone())
    generator.noise_convs[0].weight = nn.Parameter(params['decoder']['module.generator.noise_convs.0.weight'].clone())
    generator.noise_convs[0].bias = nn.Parameter(params['decoder']['module.generator.noise_convs.0.bias'].clone())
    generator.noise_convs[1].weight = nn.Parameter(params['decoder']['module.generator.noise_convs.1.weight'].clone())
    generator.noise_convs[1].bias = nn.Parameter(params['decoder']['module.generator.noise_convs.1.bias'].clone())
    for block_idx in [0, 1]:
        to_block = generator.noise_res[block_idx]
        from_prefix = f'module.generator.noise_res.{block_idx}'
        for convs_name in ['convs1', 'convs2']:
            convs_block = getattr(to_block, convs_name)
            for conv_idx in [0, 1, 2]:
                conv_layer = convs_block[conv_idx]
                conv_from_prefix = f'{from_prefix}.{convs_name}.{conv_idx}'
                conv_layer.weight_g = nn.Parameter(params['decoder'][f'{conv_from_prefix}.weight_g'].clone())
                conv_layer.weight_v = nn.Parameter(params['decoder'][f'{conv_from_prefix}.weight_v'].clone())
                conv_layer.bias = nn.Parameter(params['decoder'][f'{conv_from_prefix}.bias'].clone())

        for adain_name in ['adain1', 'adain2']:
            adain_block = getattr(to_block, adain_name)
            for adain_idx in [0, 1, 2]:
                adain_layer = adain_block[adain_idx]
                adain_from_prefix = f'{from_prefix}.{adain_name}.{adain_idx}.fc'
                adain_layer.fc.weight = nn.Parameter(params['decoder'][f'{adain_from_prefix}.weight'].clone())
                adain_layer.fc.bias = nn.Parameter(params['decoder'][f'{adain_from_prefix}.bias'].clone())

        for alpha_name in ['alpha1', 'alpha2']:
            alpha_block = getattr(to_block, alpha_name)
            for alpha_idx in [0, 1, 2]:
                alpha_from_prefix = f'{from_prefix}.{alpha_name}.{alpha_idx}'
                alpha_block[alpha_idx] = nn.Parameter(params['decoder'][alpha_from_prefix].clone())

    for ups_idx in [0, 1]:
        ups_layer = generator.ups[ups_idx]
        prefix = f'module.generator.ups.{ups_idx}'
        ups_layer.weight_g = nn.Parameter(params['decoder'][f'{prefix}.weight_g'].clone())
        ups_layer.weight_v = nn.Parameter(params['decoder'][f'{prefix}.weight_v'].clone())
        ups_layer.bias = nn.Parameter(params['decoder'][f'{prefix}.bias'].clone())
    
    for block_idx in range(6):
        res_block = generator.resblocks[block_idx]
        prefix = f'module.generator.resblocks.{block_idx}'
        
        for convs_name in ['convs1', 'convs2']:
            convs_block = getattr(res_block, convs_name)
            
            for conv_idx in range(3):
                conv_layer = convs_block[conv_idx]
                conv_prefix = f'{prefix}.{convs_name}.{conv_idx}'
                
                conv_layer.weight_g = nn.Parameter(params['decoder'][f'{conv_prefix}.weight_g'].clone())
                conv_layer.weight_v = nn.Parameter(params['decoder'][f'{conv_prefix}.weight_v'].clone())
                conv_layer.bias = nn.Parameter(params['decoder'][f'{conv_prefix}.bias'].clone())
        
        for adain_name in ['adain1', 'adain2']:
            adain_block = getattr(res_block, adain_name)
            
            for adain_idx in range(3):
                adain_layer = adain_block[adain_idx]
                adain_prefix = f'{prefix}.{adain_name}.{adain_idx}.fc'
                
                adain_layer.fc.weight = nn.Parameter(params['decoder'][f'{adain_prefix}.weight'].clone())
                adain_layer.fc.bias = nn.Parameter(params['decoder'][f'{adain_prefix}.bias'].clone())
        
        for alpha_name in ['alpha1', 'alpha2']:
            alpha_block = getattr(res_block, alpha_name)
            
            for alpha_idx in range(3):
                alpha_prefix = f'{prefix}.{alpha_name}.{alpha_idx}'
                alpha_block[alpha_idx] = nn.Parameter(params['decoder'][alpha_prefix].clone())
    
    generator.conv_post.weight_g = nn.Parameter(params['decoder']['module.generator.conv_post.weight_g'].clone())
    generator.conv_post.weight_v = nn.Parameter(params['decoder']['module.generator.conv_post.weight_v'].clone())
    generator.conv_post.bias = nn.Parameter(params['decoder']['module.generator.conv_post.bias'].clone())


def transferKokoroWeights(model, config: KokoroConfig, params):
    transferAlbertWeights(model, config, params)
    model.ff.weight = nn.Parameter(
        params['bert_encoder']['module.weight'].clone()
    )
    model.ff.bias = nn.Parameter(
        params['bert_encoder']['module.bias'].clone()
    )
    transferPredictorWeights(model, config, params)
    transferTextEncoderWeights(model, config, params)
    transferDecoderWeights(model.decoder, config, params)
    transferGeneratorWeights(model.decoder.generator, config, params)