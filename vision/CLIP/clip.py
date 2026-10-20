def clip_forward(images, texts):
    """
    images: [B, 3, H, W]
    texts : [B, L] # Token IDs

    return: scalar loss
    """

    # 1. Extract features
    # Image Encoder 和 Text Encoder
    img_feat = image_encoder(images)
    txt_feat = text_encoder(texts)

    # 2. Linear Projection: 映射到共享 Embedding Space
    img_emb = img_feat @ W_img
    txt_emb = txt_feat @ W_txt

    # 3. L2 Normalization
    # 归一化后点积等于 Cosine Similarity
    img_emb = l2_normalize(img_emb, axis=-1)
    txt_emb = l2_normalize(txt_emb, axis=-1)

    # 4. Compute Similarity Matrix
    # shape: [B, B]
    # 使用可学习的 logit_scale
    logits = img_emb @ txt_emb.T * np.exp(t)

    # 6. Symmetric Cross-Entropy
    B = logits.shape[0]
    labels = np.arange(B)
    loss_i2t = cross_entropy_manual(logits, labels)
    loss_t2i = cross_entropy_manual(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2

    return loss

def cross_entropy_manual(logits, labels):
    """
    logits: [B, B]
    labels: [B]

    return:
        loss: scalar
    """
    B = logits.shape[0]

    logits = logits - np.max(logits, axis=-1, keepdims=True)

    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    correct_probs = probs[np.arange(B), labels]

    correct_probs = np.clip(correct_probs, 1e-12, 1.0)
    loss = -np.mean(np.log(correct_probs))



    return loss