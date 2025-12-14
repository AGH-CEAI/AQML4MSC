import numpy as np
from matplotlib import pyplot as plt
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
# edges1 = cv2.Canny(X1[idx].astype(np.uint8), 50, 150)
# edges2 = cv2.Canny(X2[idx].astype(np.uint8), 50, 150)
edges1 = X1[idx] > 127
edges2 = X2[idx] > 127

reduced1 = block_reduce(edges1, block_size=2, func=np.max)
reduced2 = block_reduce(edges2, block_size=2, func=np.max)
# reduced1 = block_reduce(reduced1, block_size=(2, 2), func=np.max)
# reduced2 = block_reduce(reduced2, block_size=(2, 2), func=np.max)
# reduced1 = block_reduce(reduced1, block_size=(2, 2), func=np.mean)
# reduced2 = block_reduce(reduced2, block_size=(2, 2), func=np.mean)
vertical_projection1 = np.mean(reduced1, axis=0)
vertical_projection2 = np.mean(reduced2, axis=0)
print(vertical_projection1)
print(vertical_projection2)

print(vertical_projection1.shape)

fig, axes = plt.subplots(2, 3, figsize=(6, 9))
axes[0, 0].imshow(X1[idx], cmap="gray")
axes[0, 0].set_title("X1[%d]" % idx)
axes[0, 0].axis("off")
axes[0, 1].imshow(edges1, cmap="gray")
axes[0, 1].set_title("edges1")
axes[0, 1].axis("off")
axes[0, 2].imshow(reduced1, cmap="gray")
axes[0, 2].set_title("reduced1")
axes[0, 2].axis("off")


axes[1, 0].imshow(X2[idx], cmap="gray")
axes[1, 0].set_title("X2[%d]" % idx)
axes[1, 0].axis("off")
axes[1, 1].imshow(edges2, cmap="gray")
axes[1, 1].set_title("edges2")
axes[1, 1].axis("off")
axes[1, 2].imshow(reduced2, cmap="gray")
axes[1, 2].set_title("reduced2")
axes[1, 2].axis("off")
plt.tight_layout()
plt.show()
