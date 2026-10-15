def mmdit_block(x, c, y):
    """
    x: image tokens [B, N_img, D]
    c: text  tokens [B, N_txt, D]
    y: timestamp + pooled text condition
    """

    # 1. generate modulation parameters
    # text branch and image branch use different parameters
    text_mod = text_modulation(y)      # SiLU + linear -> [B, 6D]
    image_mod = image_modulation(y)    # SiLU + linear -> [B, 6D]

    alpha_c, beta_c, gamma_c, delta_c, eps_c, zeta_c = text_mod.chunk(6, dim=-1)
    alpha_x, beta_x, gamma_x, delta_x, eps_x, zeta_x = image_mod.chunk(6, dim=-1)

    # 2. Attention branch: LN + modulation
    c_norm = layernorm_text(c)
    x_norm = layernorm_image(x)

    c_mod = modulate(c_norm, alpha_c, beta_c)
    x_mod = modulate(x_norm, alpha_x, beta_x)

    # 3. seperate QKV projections
    q_c, k_c, v_c = text_qkv(c_mod)
    q_x, k_x, v_x = image_qkv(x_mod)

    # optional Q/K RMSNorm
    q_c = qk_norm(q_c)
    k_c = qk_norm(k_c)

    q_x = qk_norm(q_x)
    k_x = qk_norm(k_x)

    # 4. Joint Attention
    q = concat([q_c, q_x], dim=sequence)
    k = concat([k_c, k_x], dim=sequence)
    v = concat([v_c, v_x], dim=sequence)

    out = attention(q, k, v)

    # 5. Split modalities back
    out_c, out_x = split(out, [N_txt, N_img])
    
    # 6. Output projection + gated residual
    c = c + gamma_c * text_attn_proj(out_c)
    x = x + gamma_x * image_attn_proj(out_x)

    # 7. MLP branch
    c_mlp = modulate(
        layernorm_text2(c),
        delta_c,
        eps_c
    )

    x_mlp = modulate(
        layernorm_image2(x),
        delta_x,
        eps_x
    )

    c = c + zeta_c * text_mlp(c_mlp)
    x = x + zeta_x * image_mlp(x_mlp)

    return x, c