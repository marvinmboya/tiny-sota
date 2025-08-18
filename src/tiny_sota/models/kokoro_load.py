import torch.nn as nn 

def transferAlbertWeights(customalbert, config, model):
    customalbert.embedding.weight = nn.Parameter(
        model.embeddings.word_embeddings.weight.clone()
    )
    customalbert.pos_emb.weight = nn.Parameter(
        model.embeddings.position_embeddings.weight.clone()
    )
    customalbert.tok_emb.weight = nn.Parameter(
        model.embeddings.token_type_embeddings.weight.clone()
    )
    # end transfer ONE
    # transfer TWO
    customalbert.encoder.ff_in.weight = nn.Parameter(
        model.encoder.embedding_hidden_mapping_in.weight.clone()
    )
    customalbert.encoder.ff_in.bias = nn.Parameter(
        model.encoder.embedding_hidden_mapping_in.bias.clone()
    )
    customalbert.encoder.attn.Wq.weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.query.weight.clone()
    )
    customalbert.encoder.attn.Wq.bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.query.bias.clone()
    )
    customalbert.encoder.attn.Wk.weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.key.weight.clone()
    )
    customalbert.encoder.attn.Wk.bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.key.bias.clone()
    )
    customalbert.encoder.attn.Wv.weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.value.weight.clone()
    )
    customalbert.encoder.attn.Wv.bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.value.bias.clone()
    )
    customalbert.encoder.attn.Wo.weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.dense.weight.clone()
    )
    customalbert.encoder.attn.Wo.bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].attention.dense.bias.clone()
    )
    customalbert.encoder.mlp[0].weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].ffn.weight.clone()
    )
    customalbert.encoder.mlp[0].bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].ffn.bias.clone()
    )
    customalbert.encoder.mlp[2].weight = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].ffn_output.weight.clone()
    )
    customalbert.encoder.mlp[2].bias = nn.Parameter(
        model.encoder.albert_layer_groups[0].albert_layers[0].ffn_output.bias.clone()
    )
    customalbert.ff_out.weight = nn.Parameter(
        model.pooler.weight.clone()
    )
    customalbert.ff_out.bias = nn.Parameter(
        model.pooler.bias.clone()
    )

def transferKokoroWeights(model, config, params):
    transferAlbertWeights(model, config, params)