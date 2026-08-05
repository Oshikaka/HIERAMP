# HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2603.06932-b31b1b.svg)](https://arxiv.org/abs/2603.06932)&nbsp;
[![CVPR](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://arxiv.org/abs/2603.06932)

</div>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2603.06932">HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation</a>
</p>

Official implementation of **HierAmp** (CVPR 2026). Object semantics are inherently hierarchical; HierAmp matches this hierarchy with the next-scale prediction of [Visual Autoregressive Models (VAR)](https://github.com/FoundationVision/VAR). At each generation scale, a learnable **class token** identifies salient regions, and its induced activation map guides an **additive attention-logit amplification** that steers synthesis toward discriminative structures — with minimal inference overhead and no external segmenter.

## Method

At each scale $n$, a class token attends to the tokens of its own scale (scale-restricted attention). Its multi-head-averaged attention gives a semantic saliency map

$$m_n = \frac{1}{H}\sum_h \alpha^{(h)}_{n,\mathrm{cls}},\qquad a_n = \mathrm{Top}\text{-}\rho_n(m_n)\in\{0,1\}^{L_n},$$

and generation is steered by an additive logit bias applied **before** softmax:

$$\tilde{L}^{(h)}_n = L^{(h)}_n + \beta_n\,\mathbf{1}\,a_n^{\top},\qquad \tilde{\alpha}^{(h)}_n=\mathrm{softmax}(\tilde{L}^{(h)}_n).$$

At coarse scales this diversifies object layouts; at fine scales it concentrates token usage on object details. The final configuration uses $\beta_n = 5$ and $\rho_n = 50\%$ at all scales $n \in \{1,\dots,9\}$ (scale 0 contains a single patch and is excluded), applied at transformer block 14 of the depth-16 VAR.

## Setup

```bash
pip install -r requirements.txt
```

Verified with PyTorch 2.5.1 + CUDA 12.4 (any torch ≥ 2.1 should work). `flash-attn` and `xformers` are optional accelerators, not requirements.

Download the pre-trained VAR-d16 and its VQVAE (both from [FoundationVision/var](https://huggingface.co/FoundationVision/var)):

```bash
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
```

## 1. Fine-tune with class tokens (5 epochs)

The pre-trained VAR is fine-tuned for 5 epochs with a class token injected at every scale (0–9). Each class token is supervised by a lightweight shared classifier: the total loss is the VAR next-scale token cross-entropy plus $L_{cls} = \frac{1}{N}\sum_n -\log p_n(c_n^e)$ (Eq. 6, weight `--cls_lw`, default 1.0).

```bash
bash scripts/finetune.sh /path/to/imagenet 8
# → local_output/ar-ckpt-last.pth
```

The class token attends only to tokens of its own scale; its keys/values are never exposed to later scales (nor cached at inference), so the backbone's autoregressive structure is unchanged. $c_n^e$ is the last block's attention output at the class-token position of scale $n$; class-token positions are dropped before the token logits, so the token loss itself is untouched.

Notes:
- Interrupted runs auto-resume: rerunning the same command picks up `local_output/ar-ckpt-last.pth`. Clear `local_output/` (or change `--local_out_dir_path`) before starting a run with a different configuration.
- `--cls_scales=` (empty) trains a plain VAR without class tokens, e.g. for baseline comparisons.
- Progressive training (`--pg > 0`) is not supported together with class tokens.

## 2. Generate the distilled dataset

Paper configuration ($\beta=5$, $\rho=50\%$, scales 1–9, block 14):

```bash
python sample.py --var_ckpt local_output/ar-ckpt-last.pth --save_dir output/in1k_ipc10 --ipc 10
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--ipc` | 10 | images per class |
| `--beta` | `5` | amplification strength; 1 value or 3 values for the coarse/mid/fine stage-aware schedule $\beta_{1:3},\beta_{4:6},\beta_{7:9}$ |
| `--top_ratio` | `0.5` | $\rho$; 1 or 3 values, same convention |
| `--scales` | `1 … 9` | scales to amplify |
| `--layers` | `14` | transformer block(s) receiving the bias |
| `--no_amplify` | — | plain VAR baseline |
| `--classes` / `--start --end` | all 1000 | class subset (e.g. ImageNet-100 / ImageNet-Woof index lists) |
| `--cfg / --top_k / --top_p / --seed` | 4.0 / 900 / 0.95 / 0 | sampling hyperparameters |

Images are generated at 256×256 and saved at 224×224 for downstream training. See `scripts/sample.sh` for the full IPC sweep.

## 3. Evaluate

Train a student network (e.g. ResNet-18) on the generated images following standard dataset-distillation evaluation protocols with soft labels at 224×224.

## Notes for reproduction

- Amplification requires the **explicit-attention path**; it is automatically used whenever class tokens are active, even if `flash-attn`/`xformers` are installed (they still accelerate all other layers/scales at training time).
- `--cls_scales` at sampling must match the fine-tuned checkpoint (default: 0–9).
- Generation is deterministic given `--seed`; each additional batch chunk of a class uses `seed + chunk`.

## TODO

- [ ] Release the distilled dataset

## Citation

```bibtex
@inproceedings{zhao2026hieramp,
  title={HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation},
  author={Zhao, Lin and Jiang, Xinru and Xiao, Xi and Fan, Qihui and Lu, Lei and Wang, Yanzhi and Lin, Xue and Camps, Octavia and Zhao, Pu and Gu, Jianyang},
  booktitle={CVPR},
  year={2026}
}
```

## Acknowledgments

This codebase is built on [VAR](https://github.com/FoundationVision/VAR) (NeurIPS 2024 Best Paper). We thank the authors for open-sourcing it; the original license is kept in [LICENSE](LICENSE).
