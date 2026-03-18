#!/usr/bin/env python3
"""将目录下的 sample_*.png 或 latent_grid_*.png 拼成一张大图，带标号和分隔"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys

base = '/media/wpc/codebase/imageGeneration/VAE'
results_dir = sys.argv[1] if len(sys.argv) > 1 else f'{base}/results'
results_dir = results_dir if os.path.isabs(results_dir) else os.path.join(base, results_dir)
mode = sys.argv[2] if len(sys.argv) > 2 else 'sample'

if mode == 'latent_grid':
    prefix, out_name = 'latent_grid_', 'latent_grid_collage.png'
else:
    prefix, out_name = 'sample_', 'collage.png'

output_path = os.path.join(results_dir, out_name)

files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith('.png') and out_name not in f],
    key=lambda x: int(x.split('_')[-1].split('.')[0])
)

if not files:
    print(f'No {prefix}*.png files in {results_dir}')
    sys.exit(1)

imgs = [Image.open(os.path.join(results_dir, f)) for f in files]
w, h = imgs[0].size
n = len(imgs)

# 分隔与标号
gap = 16
label_h = 28
border = 2

rows = 2 if n <= 4 else (3 if n <= 9 else 4)
cols = (n + rows - 1) // rows
cell_w = w + gap
cell_h = h + label_h + gap

collage = Image.new('RGB', (cols * cell_w + gap, rows * cell_h + gap), (240, 240, 240))
draw = ImageDraw.Draw(collage)

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except OSError:
    font = ImageFont.load_default()

for i, img in enumerate(imgs):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    row, col = i // cols, i % cols
    epoch = int(files[i].split('_')[-1].split('.')[0])
    x = gap + col * cell_w
    y = gap + row * cell_h

    draw.rectangle([x, y, x + w + border * 2, y + label_h + h + border * 2],
                   fill=(255, 255, 255), outline=(180, 180, 180), width=border)
    collage.paste(img, (x + border, y + label_h + border))
    draw.text((x + border + w // 2 - 25, y + 4), f'Epoch {epoch}', fill=(60, 60, 60), font=font)

collage.save(output_path)
print(f'Saved: {output_path}')
