def double_stream_block(img, txt, temb, rope):
    """
    img:  [B, N_img, D]
    txt:  [B, N_txt, D]
    temb: [B, D]
    rope: positional info for Q/K

    return:
        img, txt
    """
    # 1. temb -> modulation parameters
    # image/text each have their own parameters
    img_shift_attn, img_scale_attn, img_gate_attn, \
    img_shift_mlp,  img_scale_mlp,  img_gate_mlp = \
        img_modulation(temb)
    
    txt_shift_attn, txt_scale_attn, txt_gate_attn, \
    txt_shift_mlp,  txt_scale_mlp,  txt_gate_mlp = \
        txt_modulation(temb)
    
    # 2. AdaLN before attention
    img_norm = layer_norm(img)
    txt_norm = layer_norm(txt)

    img_attn_in = img_norm * (1 + img_scale_attn) + img_shift_attn
    txt_attn_in = txt_norm * (1 + txt_scale_attn) + txt_shift_attn

    # 3. seperate QKV projections
    qi, ki, vi = img_qkv(img_attn_in)
    qt, kt, vt = txt_qkv(txt_attn_in)

    # 4. Concatenate text + image for joint attention
    q = concat([qt, qi], dim = token_dim)
    k = concat([kt, ki], dim = token_dim)
    v = concat([vt, vi], dim = token_dim)

    # 5. RoPE on Q & K
    q = apply_rope(q, rope)
    k = apply_rope(k, rope)

    # 6. joint self-attention
    out = attention(q, k, v)

    # split back into text/image
    txt_out, img_out = split(
        out,
        sizes=[N_txt, N_img],
        dim=token_dim
    )

    # 7. Gated residual after attn
    img = img + img_gate_attn * img_attn_proj(img_out)
    txt = txt + txt_gate_attn * txt_attn_proj(txt_out)

    # 8. AdaLN before MLP
    img_norm = layer_norm(img)
    txt_norm = layer_norm(txt)

    img_mlp_in = img_norm * (1 + img_scale_mlp) + img_shift_mlp
    txt_mlp_in = txt_norm * (1 + txt_scale_mlp) + txt_shift_mlp

    # 9. seperate FFNs + gated residual
    img = img + img_gate_mlp * img_mlp(img_mlp_in)
    txt = txt + txt_gate_mlp * txt_mlp(txt_mlp_in)

    return img, txt