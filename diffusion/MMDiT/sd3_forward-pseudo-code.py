def sd3_forward(image, caption, t, noise):
    """
    image:   RGB image
    caption: text prompt
    t:       sampled timestep
    noise:   Gaussian noise

    output:
        predicted velocity
    """

    # 1. Image -> latent
    x0 = vae_encoder(image) # x0: [B, C, H, W]

    # 2. rectified flow: construct noisy latent x_t
    # RF straight interpolation
    x_t = (1 - t) * x0 + t * noise

    # training target = noise - x0

    # Text Encoding
    # CLIP-L
    clip_l_tokens, clip_l_pooled = clip_l(caption)
    # CLIP-G
    clip_g_tokens, clip_g_pooled = clip_g(caption)
    # T5-XXL
    t5_tokens = t5(caption)

    # 4. Build global text condition c_vec
    # pooled CLIP representations
    c_vec = concat(
        [clip_l_pooled, clip_g_pooled],
        dim=channel
    )

    c_vec = pooled_text_proj(c_vec)

    # 4. Build token-level text context c_ctxt
    # CLIP token features
    clip_tokens = concat(
        [clip_l_tokens, clip_g_tokens],
        dim=channel
    )

    # 2048 -> 4096 / padding to T5 width
    clip_tokens = pad_or_project_to_t5_dim(clip_tokens)

    # align clip hidden dim with T5 hidden dim
    c_ctxt = concat(
        [clip_tokens, t5_tokens],
        dim=sequence
    )
    # c_ctxt: [B, N_txt, D]

    # 6. timestep + pooled text -> global condition y
    t_emb = sinusoidal_embedding(t)
    t_emb = timestep_mlp(t_emb)

    y = t_emb + c_vec
    # y: [B, D]

    # 7. noisy latent -> image tokens
    x = patchify(x_t) # split latent into 2x2 patches
    x = image_patch_embed(x) # Linear projection
    x = x + position_embedding
    # x: [B, N_img, D]

    # 8. MMDiT backbone
    c = c_ctxt

    for block in mmdit_blocks:
        x, c = block(
            x=x,
            c=c,
            y=y
        )
    
    # 9. Final image branch
    # only image tokens are needed for final prediction
    x = final_norm(x)

    # final adaptive modulation conditioned on y
    shift, scale = final_modulation(y).chunk(2, dim=-1)
    x = x * (1 + scale[:, None, :]) + shift[:, None, :]

    # 10. Predict patches of velocity
    x = final_linear(x)

    velocity = unpatchify(x)

    return velocity

def training_step(image, caption):
    # 1. encode clean image
    x0 = vae_encoder(image)

    # 2. sample Gaussian noise
    noise = randn_like(x0)

    # 3. sample timestep
    t = sample_timestep()
    # e.g. SD3's tailored RF timestep sampler

    # 4. RF interpolation
    x_t = (1 - t) * x0 + t * noise

    # 5. true velocity
    target = noise - x0

    # 6. MMDiT predicts velocity
    pred = sd3_transformer(
        x_t,
        caption,
        t
    )

    # flow matching loss
    loss = mse(pred, target)

    return loss