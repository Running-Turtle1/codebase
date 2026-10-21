def train_initial_prompts(clip, backlit_loader, welllit_loader):
	"""
	Stage 1: Learning Initial Prompts
    CLIP: pretrained and frozen
    Trainable: negative_prompt, positive_prompt
	"""

	# 1. Freeze CLIP
	freeze(clip.image_encoder)
	freeze(clip.text_encoder)

	# 2. Initialize learnble prompts
	negative_prompt = Parameter(rand(N, D))
	positive_prompt = Parameter(rand(N, D))

	optimizer = Adam([negative_prompt, positive_prompt])

	for img_neg, img_pos in zip(backlit_loader, welllit_loader):
		# 3. perpare images and labels
		images = concat([img_neg, img_pos], dim=0)
		labels = concat([
			zeros(len(img_neg)), # backlit: 0
			ones(len(img_pos))   # welllit: 1
		])

		# 4. Extract image features (frozen)
		with no_grad():
			img_feat = clip.image_encoder(images)

		# 5. Extract text features (gradient flows to prompts)
		prompts = stack([negative_prompt, positive_prompt])
		txt_feat = clip.text_encoder(prompts)

		# 6. L2 norm
		img_feat = normalize(img_feat, dim=-1)
		txt_feat = normalize(txt_feat, dim=-1)

		# 7. Cosine Similarity: [B, 2]
		logits = clip.logit_scale.exp() * img_feat @ txt_feat.T

		# 8. classification loss
		loss = cross_entropy(logits, labels.long())

		# 9. Update prompts ONLY
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return negative_prompt, positive_prompt


def train_enhancement_network(
	clip, clip_rn101, unet, backlit_loader,
	negative_prompt, positive_prompt
):
	"""
	Stage 2: Enhancement Network Training

    Frozen:
        CLIP ViT-B/32, CLIP RN101
        negative_prompt, positive_prompt

    Trainable:
        U-Net
	"""

	# 1. Freeze CLIP and learned prompts
	freeze(clip)
	freeze(clip_rn101)

	prompts = stack([positive_prompt, negative_prompt])
	prompts.require_grad_(False)

	with no_grad():
		txt_feat = clip.encode_soft_prompts(prompts)
		txt_feat = normalize(txt_feat, dim=-1)

	optimizer = Adam(unet.parameters(), lr=2e-5)

	for step, img_b in enumerate(backlit_loader):
		# 2. Predict illumination map
		illumination = unet(img_b) # [B, 1, H, W]

		# 3. Retinex-based enhancement
		enhanced = clamp(
			img_b / (illumination + 1e-9), 0, 1
		)

		# 4. Identity Loss (CLIP RN101 multi-layer features)
        # Input features do not need gradients
        with no_grad():
        	feat_input = clip_rn101.intermediate_features(
        		clip_preprocess(img_b)
        	)

        # Enhanced features MUST keep gradients
        feat_output = clip_rn101.intermediate_features(
        	clip_preprocess(feat_output)
        )

        # 5. Warm-up: Reconstruction Only
        if step < warm_iters:
        	weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        	loss = weighted_mse(
        		feat_output, feat_input, weights
        	) * 25
        # 6. Enhancement Training
    	else:
    		weights = [1.0, 1.0, 1.0, 1.0, 0.5]
    		loss_identity = 25 * weighted_mse(
    			feat_output, feat_input, weights
    		)

    		# CLIP image features (keep gradients)
    		img_feat = clip.image_encoder(
    			clip_preprocess(enhanced)
    		)

    		img_feat = normalize(img_feat, dim=-1)

            # Similarity: [B, 2]
            logits = 100.0 * img_feat @ txt_feat.T

            # Probability of negative/backlit class
            prob_neg = softmax(logits, dim=-1)[:, 0]

            # Minimize negative probability
            loss_clip = 320 * prob_neg.mean()

            # Total Loss
            loss = loss_clip + 0.9 * loss_identity

        # 7. Update U-Net ONLY
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return unet








