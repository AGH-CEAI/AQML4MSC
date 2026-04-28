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

# Match LLNCS fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "axes.titlesize": 12,  # LLNCS section heading
        "axes.labelsize": 10,  # LLNCS main text
    }
)


def _imshow_clean(ax, img, title=None):
    ax.imshow(img, cmap="gray")
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=15)
    ax.axis("off")


def _plot_projection(ax, projection, title=None):
    heatmap = projection[np.newaxis, :]
    ax.imshow(heatmap, aspect="equal", cmap="gray", vmin=0, vmax=1)
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=32)
    ax.axis("off")


def _plot_pair_vertical(fig, gs, row, title, img_top, img_bottom):
    ax_t = fig.add_subplot(gs[row, 0])
    ax_b = fig.add_subplot(gs[row, 1])

    _imshow_clean(ax_t, img_top, title)
    _imshow_clean(ax_b, img_bottom)

    return ax_t, ax_b


def _plot_pair_horizontal(fig, gs, col, title, img_top, img_bottom):
    ax_t = fig.add_subplot(gs[0, col])
    ax_b = fig.add_subplot(gs[1, col])

    _imshow_clean(ax_t, img_top, title)
    _imshow_clean(ax_b, img_bottom)

    return ax_t, ax_b


# ------------------------------------------------------------------
# Figure factory
# ------------------------------------------------------------------


def create_pipeline_figure(layout="vertical"):
    """
    layout ∈ {"vertical", "horizontal"}
    """

    if layout == "vertical":
        fig = plt.figure(figsize=(4.2, 7.5))
        gs = GridSpec(
            5,
            2,
            height_ratios=[2.2, 1.2, 1.2, 1.2, 0.8],
            hspace=0.35,
            wspace=0.08,
        )

        # (a) Original
        ax_a = fig.add_subplot(gs[0, :])
        _imshow_clean(ax_a, img_original, "(a) Original")

        # Processing steps
        _plot_pair_vertical(fig, gs, 1, "(b) Split (top / bottom)", img_top, img_bottom)
        _plot_pair_vertical(fig, gs, 2, "(c) Binarized", bin_top, bin_bottom)
        _plot_pair_vertical(fig, gs, 3, "(d) Downsampled", reduced_top, reduced_bottom)

        # Projection
        ax_e1 = fig.add_subplot(gs[4, 0])
        ax_e2 = fig.add_subplot(gs[4, 1])

        _plot_projection(ax_e1, vertical_projection_top, "(e) Vertical mean projection")
        _plot_projection(ax_e2, vertical_projection_bottom)

        ax_e1.text(
            0.5,
            -1.0,
            r"$x_{\mathrm{top}}$",
            transform=ax_e1.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

        ax_e2.text(
            0.5,
            -1.0,
            r"$x_{\mathrm{bottom}}$",
            transform=ax_e2.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    elif layout == "horizontal":
        fig = plt.figure(figsize=(10, 3.2))
        gs = GridSpec(
            2,
            5,
            height_ratios=[1, 1],
            width_ratios=[1.8, 1, 1, 1, 1],
            hspace=0.1,
            wspace=0.25,
        )

        # (a) Original
        ax_a = fig.add_subplot(gs[:, 0])
        _imshow_clean(ax_a, img_original, "(a) Original")

        # Processing steps
        _plot_pair_horizontal(fig, gs, 1, "(b) Split", img_top, img_bottom)
        _plot_pair_horizontal(fig, gs, 2, "(c) Binarized", bin_top, bin_bottom)
        _plot_pair_horizontal(
            fig, gs, 3, "(d) Downsampled", reduced_top, reduced_bottom
        )

        # Projection
        ax_e1 = fig.add_subplot(gs[0, 4])
        ax_e2 = fig.add_subplot(gs[1, 4])

        _plot_projection(ax_e1, vertical_projection_top, "(e) Vertical mean")
        _plot_projection(ax_e2, vertical_projection_bottom)

        ax_e1.text(
            0.5,
            -0.8,
            r"$x_{\mathrm{top}}$",
            transform=ax_e1.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

        ax_e2.text(
            0.5,
            -0.8,
            r"$x_{\mathrm{bottom}}$",
            transform=ax_e2.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    else:
        raise ValueError("layout must be 'vertical' or 'horizontal'")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Generate & save both versions
# ------------------------------------------------------------------

fig_vertical = create_pipeline_figure(layout="vertical")
fig_vertical.savefig("txt/iccs/fig/fig_vertical.pdf", backend="pgf")

fig_horizontal = create_pipeline_figure(layout="horizontal")
fig_horizontal.savefig("txt/iccs/fig/fig_horizontal.pdf", backend="pgf")

plt.show()
