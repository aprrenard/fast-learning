import numpy as np
import matplotlib.pyplot as plt
import tifffile

folder = "/mnt/lsens-analysis/Anthony_Renard/analysis_output/cicada_output/Write CI movie to tiff_2025_08_15.15-34-11"
pre_learning_tiffs = [
    "AR127_20240221_133407_baseline_wh_average.tiff",
    "AR127_20240222_152629_baseline_wh_average.tiff",
]
post_learning_tiffs = [
    "AR127_20240224_140853_baseline_wh_average.tiff",
    "AR127_20240225_142858_baseline_wh_average.tiff",
]

folder = "/mnt/lsens-analysis/Anthony_Renard/analysis_output/cicada_output/Write CI movie to tiff_2025_08_15.15-34-11"
pre_learning_tiffs = [
    "AR176_20241213_105001_baseline_wh_average.tiff",
    "AR176_20241214_145213_baseline_wh_average.tiff",
]
post_learning_tiffs = [
    "AR176_20241216_101215_baseline_wh_average.tiff",
    "AR176_20241217_132932_baseline_wh_average.tiff",
]
folder = "/mnt/lsens-analysis/Anthony_Renard/analysis_output/cicada_output/Write CI movie to tiff_2025_08_15.15-34-11"
pre_learning_tiffs = [
    "AR177_20241217_170649_baseline_wh_average.tiff",
    "AR177_20241218_145859_baseline_wh_average.tiff",
]
post_learning_tiffs = [
    "AR177_20241220_134735_baseline_wh_average.tiff",
    "AR177_20241221_121706_baseline_wh_average.tiff",
]


folder = "/mnt/lsens-analysis/Anthony_Renard/analysis_output/cicada_output/Write CI movie to tiff_2025_08_15.15-34-11"
pre_learning_tiffs = [
    "AR180_20241213_150420_baseline_wh_average.tiff",
    "AR180_20241214_194639_baseline_wh_average.tiff",
]
post_learning_tiffs = [
    "AR180_20241216_145407_baseline_wh_average.tiff",
    "AR180_20241217_160355_baseline_wh_average.tiff",
]



folder = "/mnt/lsens-analysis/Anthony_Renard/analysis_output/cicada_output/Write CI movie to tiff_2025_08_15.15-34-11"
pre_learning_tiffs = [
    "AR143_20240518_174556_baseline_wh_average.tiff",
    "AR143_20240519_141725_baseline_wh_average.tiff",
]
post_learning_tiffs = [
    "AR143_20240521_125833_baseline_wh_average.tiff",
    "AR143_20240522_172846_baseline_wh_average.tiff",
]



def load_and_average(tiff_paths, folder, frame_start, frame_end):
    images = []
    for tiff in tiff_paths:
        path = f"{folder}/{tiff}"
        with tifffile.TiffFile(path) as tif:
            data = tif.asarray()
            selected = data[frame_start:frame_end+1]
            avg = np.mean(selected, axis=0)
            images.append(avg)
    return np.mean(images, axis=0)

frame_start = 121
frame_end = 126

pre_avg = load_and_average(pre_learning_tiffs, folder, frame_start, frame_end)
post_avg = load_and_average(post_learning_tiffs, folder, frame_start, frame_end)
pre_avg = np.where(np.isinf(pre_avg), np.nan, pre_avg)
post_avg = np.where(np.isinf(post_avg), np.nan, post_avg)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(pre_avg, cmap='gray', vmin=0, vmax=np.nanmean(pre_avg)*3)
plt.title('Pre-learning Avg')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(post_avg, cmap='gray', vmin=0, vmax=np.nanmean(post_avg)*3)
plt.title('Post-learning Avg')
plt.axis('off')

plt.tight_layout()
plt.show()




# Compute all images first
all_imgs = []
for tiff in pre_learning_tiffs + post_learning_tiffs:
    img = load_and_average([tiff], folder, frame_start, frame_end)
    img = np.where(np.isinf(img), np.nan, img)
    all_imgs.append(img)

# Compute common vmin, vmax based on average intensity of all images
all_means = [np.nanmean(img) for img in all_imgs]
common_mean = np.nanmean(all_means)
vmin = 0
vmax = common_mean * 3

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Pre-learning: plot each day separately
for i, img in enumerate(all_imgs):
    im = axes[i].imshow(img, cmap='gray', vmin=0, vmax=np.nanmean(img)*3)
    axes[i].axis('off')

# Place colorbar outside the subplots, further to the right
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

for im in all_imgs:
    print(np.nanmin(im), np.nanstd(im))