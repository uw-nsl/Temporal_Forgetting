# Temporal_Forgetting

This is the official repository for "[Temporal Sampling for Forgotten Reasoning in LLMs](https://arxiv.org/pdf/2505.20196)".

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2505.20196) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- **🌟 Update**:  -->

## Abstract

Fine-tuning large language models (LLMs) is intended to improve their reasoning capabilities, yet we uncover a counterintuitive effect: models often forget how to solve problems they previously answered correctly during training. We term this phenomenon temporal forgetting and show that it is widespread across model sizes, fine-tuning methods (both Reinforcement Learning and Supervised Fine-Tuning), and multiple reasoning benchmarks. To address this gap, we introduce Temporal Sampling, a simple decoding strategy that draws outputs from multiple checkpoints along the training trajectory. This approach recovers forgotten solutions without retraining or ensembling, and leads to substantial improvements in reasoning performance, gains from 4 to 19 points in Pass@k and consistent gains in Majority@k across several benchmarks. We further extend our method to LoRA-adapted models, demonstrating that storing only adapter weights across checkpoints achieves similar benefits with minimal storage cost. By leveraging the temporal diversity inherent in training, Temporal Sampling offers a practical, compute-efficient way to surface hidden reasoning ability and rethink how we evaluate LLMs.


## Overview

![Overview](figs/teaser.jpg)



## Getting Start

We released our RL training ckpts of Qwen2.5-7B model in [Huggingface](https://huggingface.co/UWNSL).

Code of our evaluation pipeline will be released in several days :）Appreciate your patience~


## Citation
```
@article{li2025temporal,
  title={Temporal Sampling for Forgotten Reasoning in LLMs},
  author={Li, Yuetai and Xu, Zhangchen and Jiang, Fengqing and Ramasubramanian, Bhaskar and Niu, Luyao and Lin, Bill Yuchen and Yue, Xiang and Poovendran, Radha},
  journal={arXiv preprint arXiv:2505.20196},
  year={2025}
}
```