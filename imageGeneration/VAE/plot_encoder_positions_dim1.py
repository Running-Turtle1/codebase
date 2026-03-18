"""用 encoder 编码 0-9 各一个样本，将 1D 潜空间位置标在 latent_line_30.png 上"""
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from VAE import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 1
grid_path = '/media/wpc/codebase/imageGeneration/VAE/results_dim1/latent_line_30.png'
weights_path = '/media/wpc/codebase/imageGeneration/VAE/weights_dim1/vae_full_z1.pt'

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

with torch.no_grad():
    mu, _ = model.encode(x)

positions = mu.cpu().numpy().flatten()  # (10,)

# 加载 latent_line 图，z 范围 [-3, 3]
img = Image.open(grid_path)
img_arr = np.array(img)
h, w = img_arr.shape[:2]

def to_pixel(z_val):
    """z in [-3, 3] -> x pixel (水平方向)"""
    px = (z_val + 3) / 6 * w
    return int(px)

fig, ax = plt.subplots(figsize=(12, 2))
if len(img_arr.shape) == 2:
    ax.imshow(img_arr, cmap='Greys_r', aspect='auto')
else:
    ax.imshow(img_arr, aspect='auto')

# 1D：点在底部，y 固定
py = h - 15
for i, z_val in enumerate(positions):
    px = to_pixel(z_val)
    ax.scatter(px, py, s=280, c='red', edgecolors='white', linewidths=3, zorder=5)
    ax.annotate(str(labels[i]), (px, py), xytext=(0, -25), textcoords='offset points',
                fontsize=14, color='red', weight='bold', ha='center', va='top', zorder=6)

ax.axis('off')
plt.tight_layout()
plt.savefig(grid_path, bbox_inches='tight', dpi=100)
plt.close()
print(f'Saved: {grid_path}')
print('Encoder positions (digit, z):')
for d, z in zip(labels, positions):
    print(f'  {d}: {z:.2f}')
