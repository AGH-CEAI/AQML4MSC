from aqml4msc.metrics import corrected_ttest_one_tailed

n_train = 230
n_test = 58

results = {
    "mlp_tab": [0.966, 0.948, 0.914, 0.947, 0.947],
    "mlp_images": [0.862, 0.897, 0.948, 0.842, 0.842],
    "mlp_multimodal": [0.914, 1.000, 0.983, 0.965, 1.000],
    "mlp_multimodal_small": [0.931, 0.983, 0.983, 0.983, 1.000],
    "qnn_basic_tab": [0.776, 0.862, 0.862, 0.825, 0.544],
    "qnn_basic_images": [0.741, 0.707, 0.862, 0.772, 0.772],
    "qnn_basic_multimodal": [0.724, 0.845, 0.879, 0.772, 0.842],
    "qnn_basic_multimodal_small": [0.759, 0.759, 0.862, 0.737, 0.807],
    "qnn_strongly_tab": [0.741, 0.707, 0.724, 0.737, 0.719],
    "qnn_strongly_images": [0.690, 0.690, 0.707, 0.719, 0.702],
    "qnn_strongly_multimodal": [0.690, 0.724, 0.741, 0.719, 0.666],
    "qnn_strongly_multimodal_small": [0.724, 0.690, 0.759, 0.737, 0.737],
    "qnn_strongly_probs_tab": [0.828, 0.828, 0.845, 0.719, 0.754],
    "qnn_strongly_probs_images": [0.707, 0.724, 0.741, 0.737, 0.719],
    "qnn_strongly_probs_multimodal": [0.741, 0.776, 0.776, 0.754, 0.754],
    "qnn_strongly_probs_multimodal_small": [0.810, 0.862, 0.845, 0.825, 0.772],
    "qnn_strongly_probs_multimodal_bilinear": [0.776, 0.914, 0.862, 0.895, 0.860],
    "qnn_strongly_probs_multimodal_bilinear_small": [0.828, 0.931, 0.897, 0.877, 0.877],
}

fight_list = [
    ["mlp_multimodal", "mlp_tab"],
    ["mlp_multimodal", "mlp_images"],
    ["mlp_multimodal_small", "mlp_tab"],
    ["mlp_multimodal_small", "mlp_images"],
    ["qnn_basic_multimodal", "qnn_basic_tab"],
    ["qnn_basic_multimodal", "qnn_basic_images"],
    ["qnn_basic_multimodal_small", "qnn_basic_tab"],
    ["qnn_basic_multimodal_small", "qnn_basic_images"],
    ["qnn_strongly_multimodal", "qnn_strongly_tab"],
    ["qnn_strongly_multimodal", "qnn_strongly_images"],
    ["qnn_strongly_multimodal_small", "qnn_strongly_tab"],
    ["qnn_strongly_multimodal_small", "qnn_strongly_images"],
    ["qnn_strongly_probs_multimodal", "qnn_strongly_probs_tab"],
    ["qnn_strongly_probs_multimodal", "qnn_strongly_probs_images"],
    ["qnn_strongly_probs_multimodal_small", "qnn_strongly_probs_tab"],
    ["qnn_strongly_probs_multimodal_small", "qnn_strongly_probs_images"],
    ["qnn_strongly_probs_multimodal_bilinear", "qnn_strongly_probs_tab"],
    ["qnn_strongly_probs_multimodal_bilinear", "qnn_strongly_probs_images"],
    ["qnn_strongly_probs_multimodal_bilinear_small", "qnn_strongly_probs_tab"],
    ["qnn_strongly_probs_multimodal_bilinear_small", "qnn_strongly_probs_images"],
]

for res1, res2 in fight_list:
    t, p = corrected_ttest_one_tailed(
        results[res1],
        results[res2],
        n_train=n_train,
        n_test=n_test,
        alternative="greater",
    )
    print(f"{res1} vs {res2}: t={t:.3f}, p={p:.3f}")
