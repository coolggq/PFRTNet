import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非图形界面的后端
import matplotlib.pyplot as plt
from skimage import io, color

# ============================== 参数设置 ==============================
original_img_path = "/home/gegq/Change_CMF/saved_images/pdfs_image.png"    # 替换为原始图像路径
reconstructed_img_path = "/home/gegq/Change_CMF/saved_images/complement1.png" # 替换为重建图像路径
output_dir = "/home/gegq/Change_CMF/saved_images"                # 输出结果保存路径
# ====================================================================

def load_and_preprocess(img_path):
    """加载图像并预处理为归一化灰度图"""
    img = io.imread(img_path)
    if img.ndim == 3:  # 转换为单通道灰度图
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
    return img

# 加载图像
original = load_and_preprocess(original_img_path)
reconstructed = load_and_preprocess(reconstructed_img_path)

# 确保图像尺寸一致
assert original.shape == reconstructed.shape, "图像尺寸不一致！"

# ========================= 1. 差异图生成 =========================
diff_map = np.abs(original - reconstructed)  # 计算绝对差异
diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())  # 归一化到[0,1]
# 设置误差区间范围，例如0.2到0.25之间的误差设为0
# lower_threshold = 0.25
# upper_threshold = 0.4
# diff_map[(diff_map >= lower_threshold) & (diff_map <= upper_threshold)] = 0
# 可视化设置
plt.figure(figsize=(8, 8))  # 设置为仅显示差异图

# 差异图（热图）
heatmap = plt.imshow(diff_map, cmap='jet')
plt.colorbar(heatmap, fraction=0.046, pad=0.04)  # 添加颜色条
# plt.title("Difference Map (Absolute Error)")
plt.axis('off')

# 保存结果
plt.tight_layout()  # 自动调整布局
plt.savefig(f"{output_dir}/difference_map.png", dpi=300, bbox_inches='tight')
plt.show()
