from typing import List

import torchvision
import torch

with open("imagenet1000_clsidx_to_labels.txt") as f:
    idx2label = eval(f.read())


def get_label_text(idx: int) -> str:
    return idx2label[idx]


def get_image(img_tensor, de_normalize=False, data_config=None):
    """
    Inverses the normalization in the transformation; could possibly ignore other transforms (if exists).
    """
    if de_normalize:
        assert data_config is not None
        inv_normalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0., 0., 0.],
                std=1 / torch.tensor(data_config['std'])),
            torchvision.transforms.Normalize(
                mean=-1 * torch.tensor(data_config['mean']),
                std=[1., 1., 1.]),
        ])
        img_tensor = inv_normalize(img_tensor)

    if img_tensor.shape[0] == 1:  # hack
        img_tensor = img_tensor[0]

    return torchvision.transforms.ToPILImage()(img_tensor)


def get_logits(model, norm_func, img_tensor, normalize_before=True):
    if normalize_before:
        img_tensor = norm_func(img_tensor)
    return model(img_tensor)


def visualize_classification(logits: torch.Tensor, additional_labels: List[int] = []):
    import matplotlib.pyplot as plt

    logits = logits.cpu()
    probabilities = (logits.softmax(dim=-1) * 100)[0]

    top_probabilities, top_class_indices = torch.sort(probabilities, descending=True)

    # Get top 5 classes and always include CAT_IDX and DOG_IDX if not in top 5
    indices_to_show = []
    probs_to_show = []
    labels_to_show = []

    for rank, (idx, prob) in enumerate(zip(top_class_indices, top_probabilities)):
        if rank < 5 or idx in additional_labels:
            indices_to_show.append(idx.item())
            probs_to_show.append(prob.item())
            labels_to_show.append(get_label_text(idx.item()))

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(3, len(indices_to_show) * 0.4)))

    bars = ax.barh(labels_to_show, probs_to_show, color='skyblue')

    # Add percentage labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(min(width + 1, 95), bar.get_y() + bar.get_height() / 2,
                f'{probs_to_show[i]:.2f}%',
                va='center')

    # Customize the plot
    ax.set_xlabel('Probability (%)')
    ax.set_title('Classification Results')

    # Set x-axis limit to 100%
    ax.set_xlim(0, 100)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig
