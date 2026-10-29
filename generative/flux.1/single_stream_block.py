def single_stream_block(x, temb, rope):
    """
    x: [B, N_txt+N_img, D]
    """

    # 1. modulation
    shift, scale, gate = modulate(temb)

    # 2. AdaLN
    h = layer_norm(x)
    h = h * (1 + scale) + shift

    # 3. attention branch
    q, k, v = qkv(h)

    q = q_norm(q)
    k = k_norm(k)

    q, k = apply_rope(q, k, rope)

    attn_out = attention(q, k, v)

    # 4. MLP branch (parallel)
    mlp_out = mlp(h)

    # 5. fuse
    out = concat([attn_out, mlp_out], dim=-1)
    out = output_linear(out)

    # 6. gated residual
    x = x + gate * out

    return x