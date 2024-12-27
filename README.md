# Demonstrating PGD

Demonstrating a targeted, inference-time, adversarial example attack on state-of-the-art image classifiers.

[[Colab Notebook]]()

### Notes
- Targets the 1st ranked model in [`timm`'s leaderboard](https://huggingface.co/spaces/timm/leaderboard).
- Generally, targets classifiers trained on `ImageNet-1K` dataset, 
and fits to other models loaded with [`timm`](https://github.com/huggingface/pytorch-image-models) framework. 