"""用 encoder 编码 0-9 各一个样本，将潜空间位置标在 latent_grid_30.png 上"""
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from VAE import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 2
grid_path = '/media/wpc/codebase/imageGeneration/VAE/results_dim2/latent_grid_30.png'
weights_path = '/media/wpc/codebase/imageGeneration/VAE/weights_dim2/vae_full_z2.pt'

# 加载模型
model = VAE(z_dim=z_dim).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 从 MNIST 取 0-9 各一个样本
mnist = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
digit_to_idx = {d: [] for d in range(10)}
for i, (_, label) in enumerate(mnist):
    digit_to_idx[label].append(i)
    if all(len(v) >= 1 for v in digit_to_idx.values()):
        break

samples = []
labels = []
for d in range(10):
    idx = digit_to_idx[d][0]
    img, _ = mnist[idx]
    samples.append(img)
    labels.append(d)

x = torch.stack(samples).to(device).view(-1, 784)

# encoder 得到潜空间位置 mu
with torch.no_grad():
    mu, _ = model.encode(x)

positions = mu.cpu().numpy()  # (10, 2)

# 加载 latent_grid 图，坐标范围 [-3, 3] x [-3, 3]
img = Image.open(grid_path)
img_arr = np.array(img)
h, w = img_arr.shape[:2]

# 将 (z_x, z_y) 映射到像素坐标
def to_pixel(zx, zy):
    px = (zx + 3) / 6 * w
    py = (zy + 3) / 6 * h  # 图像 y 向下
    return int(px), int(py)

# 在图上标点
fig, ax = plt.subplots(figsize=(10, 10))
if len(img_arr.shape) == 2:
    ax.imshow(img_arr, cmap='Greys_r')
else:
    ax.imshow(img_arr)

for i, (zx, zy) in enumerate(positions):
    px, py = to_pixel(zx, zy)
    ax.scatter(px, py, s=280, c='red', edgecolors='white', linewidths=3, zorder=5)
    # 数字标在红点右上方，避免被遮挡
    ax.annotate(str(labels[i]), (px, py), xytext=(22, 22), textcoords='offset points',
                fontsize=16, color='red', weight='bold', ha='center', va='center', zorder=6)

ax.axis('off')
plt.tight_layout()
plt.savefig(grid_path, bbox_inches='tight', dpi=100)
plt.close()
print(f'Saved: {grid_path}')
print('Encoder positions (digit, z_x, z_y):')
for d, (zx, zy) in zip(labels, positions):
    print(f'  {d}: ({zx:.2f}, {zy:.2f})')
