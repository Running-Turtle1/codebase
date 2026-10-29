# n_avg=1 的简化版本

z_edit = source_image

for t in timesteps: # large -> small
	noise = randn_like(source_image)

	# 1. construct source latent
	z_src = (1 - t) * source_image + t * noise

	# 2. construct target latent
	z_tar = z_edit + z_src - source_image

	# 3. predict velocities
	v_src = model(z_src, t, source_prompt)
	v_tar = model(z_tar, t, target_prompt)

	# 4. Differential editing 
	v_edit = v_tar - v_src

	# 5. Update edited latent
    z_edit = z_edit + dt * v_edit  # dt < 0