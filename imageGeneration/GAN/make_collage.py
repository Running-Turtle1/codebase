#!/usr/bin/env python3
"""将 results_gan 下的 sample_*.png 和 final.png 拼成一张大图"""
from PIL import Image, ImageDraw, ImageFont
import os

results_dir = '/media/wpc/codebase/imageGeneration/GAN/results_gan'
output_path = os.path.join(results_dir, 'collage.png')

# sample_1, sample_10, ... 按数字排序，最后加 final
sample_files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith('sample_') and f.endswith('.png')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)
files = sample_files + (['final.png'] if os.path.exists(os.path.join(results_dir, 'final.png')) else [])

if not files:
    print('No images found')
    exit(1)

imgs = [Image.open(os.path.join(results_dir, f)) for f in files]
w, h = imgs[0].size
n = len(files)

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
    label = 'Final' if files[i] == 'final.png' else f'Epoch {files[i].split("_")[1].split(".")[0]}'
    x = gap + col * cell_w
    y = gap + row * cell_h

    draw.rectangle([x, y, x + w + border * 2, y + label_h + h + border * 2],
                   fill=(255, 255, 255), outline=(180, 180, 180), width=border)
    collage.paste(img, (x + border, y + label_h + border))
    tw = len(label) * 8
    draw.text((x + border + w // 2 - tw // 2, y + 4), label, fill=(60, 60, 60), font=font)

collage.save(output_path)
print(f'Saved: {output_path}')
