# Attacking Image Classifiers

Demonstrating an inference-time adversarial-example attack on state-of-the-art image classifiers.

[[Blogpost]](https://matanbt.github.io/post/pgdemo2024/)  |  [[Colab Notebook]](https://colab.research.google.com/github/matanbt/demo-pgd/blob/main/demo-pgd.ipynb)

### Notes
- All logic is implemented from scratch and self-contained in the [notebook](./demo-pgd.ipynb).
- Targets the 1st ranked model in [`timm`'s leaderboard](https://huggingface.co/spaces/timm/leaderboard).
- Generally, targets classifiers trained on `ImageNet-1K` dataset, 
and fits to other models loaded with [`timm`](https://github.com/huggingface/pytorch-image-models) framework. 
