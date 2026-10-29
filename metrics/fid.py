def compute_fid(real_images, generated_images):
	"""
    real_images: [N, 3, H, W]
    generated_images: [M, 3, H, W]
    return: FID (lower is better)
    """

    # 1. Load pretrained Inception-v3
    inception = load_inception_v3()
    inception.eval()
    freeze(inception)

    # 2. Extract features
    # [N, 2048], [M, 2048]
    real_feat = inception_features(real_images)
    gen_feat = inception_features(generated_images)

    # 3. Compute Gaussian statistics
    mu_r = mean(real_feat, dim=0) # [2048]
    mu_g = mean(gen_feat, dim=0)  # [2048]

    cov_r = covariance(real_feat) # [2048, 2048]
    cov_g = covariance(gen_feat)  # [2048, 2048]

    # 4. Compute Fréchet distance
    mean_diff = sum((mu_r - mu_g) ** 2)

    # Matrix square root, NOT element-wise sqrt
    cov_mean = matrix_sqrt(cov_r @ cov_g)

    fid = mean_diff + trace(
    	cov_r + cov_g - 2 * cov_mean
    )

    return fid
