import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.measure import block_reduce

train_val_images = "data/train_images.npy"
train_val_labels = "data/train_labels.npy"
test_images = "data/test_images.npy"
test_labels = "data/test_labels.npy"

train_val_images = np.load(train_val_images)
train_val_labels = np.load(train_val_labels)

test_images = np.load(test_images)
test_labels = np.load(test_labels)

print(test_images.shape)  # (10000, 28, 28)
print(train_val_images.shape)  # (60000, 28, 28)


# Split images in half
X1 = train_val_images[:, :14, :]
X2 = train_val_images[:, 14:, :]

# plot both pieces
# show 2x2: left/top original half + its edges, right/bottom other half + its edges
idx = 100
img_original = train_val_images[idx]

img_top = X1[idx]
img_bottom = X2[idx]

bin_top = img_top > 127
bin_bottom = img_bottom > 127

reduced_top = block_reduce(bin_top, block_size=2, func=np.max)
reduced_bottom = block_reduce(bin_bottom, block_size=2, func=np.max)

vertical_projection_top = np.mean(reduced_top, axis=0)
vertical_projection_bottom = np.mean(reduced_bottom, axis=0)
print(vertical_projection_top)
print(vertical_projection_bottom)

print(vertical_projection_top.shape)


# Visualize preprocessing pipeline on example


fig = plt.figure(figsize=(4.2, 7.5))  # single-column friendly

gs = GridSpec(5, 2, height_ratios=[2.2, 1.2, 1.2, 1.2, 0.8], hspace=0.35, wspace=0.08)

# (a) Original (span both columns)
ax_a = fig.add_subplot(gs[0, :])
ax_a.imshow(img_original, cmap="gray")
ax_a.set_title("(a) Original", loc="left", fontsize=11, pad=5)
ax_a.axis("off")


# Helper function for paired rows
def plot_pair(row, title, img_top, img_bottom):
    ax_t = fig.add_subplot(gs[row, 0])
    ax_b = fig.add_subplot(gs[row, 1])

    ax_t.imshow(img_top, cmap="gray")
    ax_b.imshow(img_bottom, cmap="gray")

    ax_t.set_title(title, loc="left", fontsize=11, pad=5)
    ax_t.axis("off")
    ax_b.axis("off")

    return ax_t, ax_b


# (b) Split
plot_pair(1, "(b) Split (top / bottom)", img_top, img_bottom)

# (c) Binarized
plot_pair(2, "(c) Binarized", bin_top, bin_bottom)

# (d) Downsampled
plot_pair(3, "(d) Downsampled", reduced_top, reduced_bottom)


# (e) Vertical projection (annotated heatmap tiles)

ax_e1 = fig.add_subplot(gs[4, 0])
ax_e2 = fig.add_subplot(gs[4, 1])

for ax, heatmap, label in [
    (ax_e1, vertical_projection_top, "(e) Vertical mean projection"),
    (ax_e2, vertical_projection_bottom, None),
]:
    heatmap = heatmap[np.newaxis, :]
    ax.imshow(heatmap, aspect="equal", cmap="gray", vmin=0, vmax=1)

    # for i, v in enumerate(heatmap[0]):
    #     ax.text(i, 0, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white")

    ax.axis("off")
    if label:
        ax.set_title(label, loc="left", fontsize=11, pad=5)

plt.tight_layout()
# plt.savefig("preprocessing_pipeline.pdf", bbox_inches="tight", dpi=300)
plt.show()
