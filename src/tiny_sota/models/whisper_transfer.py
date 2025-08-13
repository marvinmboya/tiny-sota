from .tiny_load import assign 
from .configs import Whisper_Small

def transferWhisperWeights(model, config: Whisper_Small, params):
    encoder = model.encoder
    # ENCODER CONVS
    encoder.conv1.weight = assign(encoder.conv1.weight, 
        params["encoder.conv1.weight"], "encoder.conv1.weight")
    encoder.conv1.bias = assign(encoder.conv1.bias, 
        params["encoder.conv1.bias"], "encoder.conv1.bias")
    encoder.conv2.weight = assign(encoder.conv2.weight, 
        params["encoder.conv2.weight"], "encoder.conv2.weight")
    encoder.conv2.bias = assign(encoder.conv2.bias, 
        params["encoder.conv2.bias"], "encoder.conv2.bias")
    # ENCODER POS EMBEDS
    encoder.pos_emb = assign(
        encoder.pos_emb, 
        params["encoder.positional_embedding"], 
        "encoder.positional_embedding"
    )
    # ENCODER ASSIGN
    for i in range(config.n_audio_layer):
        block = encoder.blocks[i]
        # attention params
        block.attn.Wq.weight = assign(block.attn.Wq.weight, 
        params[f"encoder.blocks.{i}.attn.query.weight"], f"encoder.blocks.{i}.attn.query.weight")
        block.attn.Wq.bias = assign(block.attn.Wq.bias, 
        params[f"encoder.blocks.{i}.attn.query.bias"], f"encoder.blocks.{i}.attn.query.bias")
        block.attn.Wk.weight = assign(block.attn.Wk.weight, 
        params[f"encoder.blocks.{i}.attn.key.weight"], 
        f"encoder.blocks.{i}.attn.key.weight")
        block.attn.Wv.weight = assign(block.attn.Wv.weight, 
        params[f"encoder.blocks.{i}.attn.value.weight"], f"encoder.blocks.{i}.attn.value.weight")
        block.attn.Wv.bias = assign(block.attn.Wv.bias, 
        params[f"encoder.blocks.{i}.attn.value.bias"], f"encoder.blocks.{i}.attn.value.bias")
        block.attn.Wo.weight = assign(block.attn.Wo.weight, 
        params[f"encoder.blocks.{i}.attn.out.weight"], f"encoder.blocks.{i}.attn.out.weight")
        block.attn.Wo.bias = assign(block.attn.Wo.bias, 
        params[f"encoder.blocks.{i}.attn.out.bias"], f"encoder.blocks.{i}.attn.out.bias")
        # layernorm params
        block.ln.weight = assign(block.ln.weight, 
        params[f"encoder.blocks.{i}.attn_ln.weight"], f"encoder.blocks.{i}.attn_ln.weight")
        block.ln.bias = assign(block.ln.bias, 
        params[f"encoder.blocks.{i}.attn_ln.bias"], f"encoder.blocks.{i}.attn_ln.bias")
        block.ff_ln.weight = assign(block.ff_ln.weight, 
        params[f"encoder.blocks.{i}.mlp_ln.weight"], f"encoder.blocks.{i}.mlp_ln.weight")
        block.ff_ln.bias = assign(block.ff_ln.bias, 
        params[f"encoder.blocks.{i}.mlp_ln.bias"], f"encoder.blocks.{i}.mlp_ln.bias")
        # mlp params
        block.feed_forward[0].weight = assign(block.feed_forward[0].weight, 
        params[f"encoder.blocks.{i}.mlp.0.weight"], f"encoder.blocks.{i}.mlp.0.weight")
        block.feed_forward[0].bias = assign(block.feed_forward[0].bias, 
        params[f"encoder.blocks.{i}.mlp.0.bias"], f"encoder.blocks.{i}.mlp.0.bias")
        block.feed_forward[2].weight = assign(block.feed_forward[2].weight, 
        params[f"encoder.blocks.{i}.mlp.2.weight"], f"encoder.blocks.{i}.mlp.2.weight")
        block.feed_forward[2].bias = assign(block.feed_forward[2].bias, 
        params[f"encoder.blocks.{i}.mlp.2.bias"], f"encoder.blocks.{i}.mlp.2.bias")
    # ENCODER POST LN 
    encoder.ln.weight = assign(encoder.ln.weight, 
    params["encoder.ln_post.weight"], "encoder.ln_post.weight")
    encoder.ln.bias = assign(encoder.ln.bias, 
    params["encoder.ln_post.bias"], "encoder.ln_post.bias")

    # DECODER ASSIGN
    decoder = model.decoder
    # DECODER POS EMBEDS
    decoder.pos_emb = assign(
        decoder.pos_emb, 
        params["decoder.positional_embedding"], 
        "decoder.positional_embedding"
    )
    decoder.token_emb.weight = assign(
        decoder.token_emb.weight, 
        params["decoder.token_embedding.weight"], 
        "decoder.token_embedding.weight"
    )
    for i in range(config.n_text_layer):
        block = decoder.blocks[i]
        # attention params
        block.attn.Wq.weight = assign(block.attn.Wq.weight, 
        params[f"decoder.blocks.{i}.attn.query.weight"], f"decoder.blocks.{i}.attn.query.weight")
        block.attn.Wq.bias = assign(block.attn.Wq.bias, 
        params[f"decoder.blocks.{i}.attn.query.bias"], f"decoder.blocks.{i}.attn.query.bias")
        block.attn.Wk.weight = assign(block.attn.Wk.weight, 
        params[f"decoder.blocks.{i}.attn.key.weight"], f"decoder.blocks.{i}.attn.key.weight")
        block.attn.Wv.weight = assign(block.attn.Wv.weight, 
        params[f"decoder.blocks.{i}.attn.value.weight"], f"decoder.blocks.{i}.attn.value.weight")
        block.attn.Wv.bias = assign(block.attn.Wv.bias, 
        params[f"decoder.blocks.{i}.attn.value.bias"], f"decoder.blocks.{i}.attn.value.bias")
        block.attn.Wo.weight = assign(block.attn.Wo.weight, 
        params[f"decoder.blocks.{i}.attn.out.weight"], f"decoder.blocks.{i}.attn.out.weight")
        block.attn.Wo.bias = assign(block.attn.Wo.bias, 
        params[f"decoder.blocks.{i}.attn.out.bias"], f"decoder.blocks.{i}.attn.out.bias")
        # cross attention params
        block.cross_attn.Wq.weight = assign(block.cross_attn.Wq.weight, 
        params[f"decoder.blocks.{i}.cross_attn.query.weight"], f"decoder.blocks.{i}.cross_attn.query.weight")
        block.cross_attn.Wq.bias = assign(block.cross_attn.Wq.bias, 
        params[f"decoder.blocks.{i}.cross_attn.query.bias"], f"decoder.blocks.{i}.cross_attn.query.bias")
        block.cross_attn.Wk.weight = assign(block.cross_attn.Wk.weight, 
        params[f"decoder.blocks.{i}.cross_attn.key.weight"], f"decoder.blocks.{i}.cross_attn.key.weight")
        block.cross_attn.Wv.weight = assign(block.cross_attn.Wv.weight, 
        params[f"decoder.blocks.{i}.cross_attn.value.weight"], f"decoder.blocks.{i}.cross_attn.value.weight")
        block.cross_attn.Wv.bias = assign(block.cross_attn.Wv.bias, 
        params[f"decoder.blocks.{i}.cross_attn.value.bias"], f"decoder.blocks.{i}.cross_attn.value.bias")
        block.cross_attn.Wo.weight = assign(block.cross_attn.Wo.weight, 
        params[f"decoder.blocks.{i}.cross_attn.out.weight"], f"decoder.blocks.{i}.cross_attn.out.weight")
        block.cross_attn.Wo.bias = assign(block.cross_attn.Wo.bias, 
        params[f"decoder.blocks.{i}.cross_attn.out.bias"], f"decoder.blocks.{i}.cross_attn.out.bias")
        # layernorm params
        block.ln.weight = assign(block.ln.weight, 
        params[f"decoder.blocks.{i}.attn_ln.weight"], f"decoder.blocks.{i}.attn_ln.weight")
        block.ln.bias = assign(block.ln.bias, 
        params[f"decoder.blocks.{i}.attn_ln.bias"], f"decoder.blocks.{i}.attn_ln.bias")
        block.ff_ln.weight = assign(block.ff_ln.weight, 
        params[f"decoder.blocks.{i}.mlp_ln.weight"], f"decoder.blocks.{i}.mlp_ln.weight")
        block.ff_ln.bias = assign(block.ff_ln.bias, 
        params[f"decoder.blocks.{i}.mlp_ln.bias"], f"decoder.blocks.{i}.mlp_ln.bias")
        # cross layernorm params
        block.cross_ln.weight = assign(block.cross_ln.weight, 
        params[f"decoder.blocks.{i}.cross_attn_ln.weight"], f"decoder.blocks.{i}.cross_attn_ln.weight")
        block.cross_ln.bias = assign(block.cross_ln.bias, 
        params[f"decoder.blocks.{i}.cross_attn_ln.bias"], f"decoder.blocks.{i}.cross_attn_ln.bias")
        # mlp params
        block.feed_forward[0].weight = assign(block.feed_forward[0].weight, 
        params[f"decoder.blocks.{i}.mlp.0.weight"], f"decoder.blocks.{i}.mlp.0.weight")
        block.feed_forward[0].bias = assign(block.feed_forward[0].bias, 
        params[f"decoder.blocks.{i}.mlp.0.bias"], f"decoder.blocks.{i}.mlp.0.bias")
        block.feed_forward[2].weight = assign(block.feed_forward[2].weight, 
        params[f"decoder.blocks.{i}.mlp.2.weight"], f"decoder.blocks.{i}.mlp.2.weight")
        block.feed_forward[2].bias = assign(block.feed_forward[2].bias, 
        params[f"decoder.blocks.{i}.mlp.2.bias"], f"decoder.blocks.{i}.mlp.2.bias")



    
    
