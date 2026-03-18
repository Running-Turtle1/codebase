#!/usr/bin/env python3
"""将 results 目录下的 sample_*.png 拼成一张大图，带标号和分隔"""
from PIL import Image, ImageDraw, ImageFont
import os

results_dir = '/media/wpc/codebase/imageGeneration/VAE/results'
output_path = '/media/wpc/codebase/imageGeneration/VAE/results/collage.png'

files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith('sample_') and f.endswith('.png') and f != 'collage.png'],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

imgs = [Image.open(os.path.join(results_dir, f)) for f in files]
w, h = imgs[0].size
n = len(imgs)

# 分隔与标号
gap = 16          # 图片间距
label_h = 28      # 标号区域高度
border = 2        # 分隔线粗细

rows, cols = 3, 5
cell_w = w + gap
cell_h = h + label_h + gap

collage = Image.new('RGB', (cols * cell_w + gap, rows * cell_h + gap), (240, 240, 240))
draw = ImageDraw.Draw(collage)

# 尝试加载字体
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except OSError:
    font = ImageFont.load_default()

for i, img in enumerate(imgs):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    row, col = i // cols, i % cols
    epoch = int(files[i].split('_')[1].split('.')[0])
    x = gap + col * cell_w
    y = gap + row * cell_h

    # 白底 + 灰边框
    draw.rectangle([x, y, x + w + border * 2, y + label_h + h + border * 2],
                   fill=(255, 255, 255), outline=(180, 180, 180), width=border)
    # 贴图
    collage.paste(img, (x + border, y + label_h + border))
    # 标号 "Epoch 1"
    draw.text((x + border + w // 2 - 25, y + 4), f'Epoch {epoch}', fill=(60, 60, 60), font=font)

collage.save(output_path)
print(f'Saved: {output_path}')
