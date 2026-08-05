"""
HierAmp: generate a distilled dataset with coarse-to-fine semantic amplification.

Example (paper configuration: beta=5, rho=50%, scales 1-9):
    python sample.py --var_ckpt checkpoints/hieramp_d16.pth --save_dir output/in1k_ipc10 --ipc 10
"""
import argparse
import json
import os
import os.path as osp
import random

import numpy as np
import PIL.Image as PImage
import torch

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import build_vae_var

HF_HOME = 'https://huggingface.co/FoundationVision/var/resolve/main'


def expand_stagewise(values, patch_nums, name):
    """Expand a 1-value (uniform) or 3-value (coarse/mid/fine stage-aware) schedule to a per-scale list."""
    n = len(patch_nums)
    if len(values) == 1:
        return [float(values[0])] * n
    if len(values) == 3:   # stages: coarse (1-3), mid (4-6), fine (7-9); scale 0 is never amplified
        out = [float(values[0])] * n
        for si in range(n):
            out[si] = float(values[0]) if si <= 3 else float(values[1]) if si <= 6 else float(values[2])
        return out
    raise ValueError(f'--{name} expects 1 value (uniform) or 3 values (coarse/mid/fine), got {len(values)}')


def main():
    parser = argparse.ArgumentParser(description='HierAmp dataset distillation sampling')
    # model
    parser.add_argument('--var_ckpt', type=str, required=True, help='VAR checkpoint fine-tuned with class tokens (see train.py)')
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth', help='VQVAE checkpoint (auto-downloaded if absent)')
    parser.add_argument('--depth', type=int, default=16, choices=[16, 20, 24, 30])
    parser.add_argument('--cls_scales', type=int, nargs='+', default=list(range(0, 10)),
                        help='scales carrying a class token; must match the fine-tuned checkpoint')
    parser.add_argument('--no_amplify', action='store_true', help='disable amplification (plain VAR baseline)')
    parser.add_argument('--beta', type=float, nargs='+', default=[5.0],
                        help='amplification strength; 1 value (uniform) or 3 values (coarse/mid/fine stages)')
    parser.add_argument('--top_ratio', type=float, nargs='+', default=[0.5],
                        help='rho, ratio of top salient positions to amplify; 1 or 3 values')
    parser.add_argument('--scales', type=int, nargs='+', default=list(range(1, 10)),
                        help='scales to amplify (scale 0 has a single patch and is excluded)')
    parser.add_argument('--layers', type=int, nargs='+', default=[14],
                        help='transformer block indices where the logit bias is applied')
    # sampling
    parser.add_argument('--cfg', type=float, default=4.0, help='classifier-free guidance ratio')
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    # dataset distillation
    parser.add_argument('--ipc', type=int, default=10, help='images per class')
    parser.add_argument('--batch_size', type=int, default=25, help='max images generated per forward pass (per CFG half)')
    parser.add_argument('--start', type=int, default=0, help='first class index (inclusive)')
    parser.add_argument('--end', type=int, default=1000, help='last class index (exclusive)')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='explicit class indices (overrides --start/--end)')
    parser.add_argument('--mapping_file', type=str, default='imagenet_1k_mapping.json')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--save_reso', type=int, default=224, help='resolution of saved images (generated at 256)')
    args = parser.parse_args()

    beta = expand_stagewise(args.beta, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), name='beta')
    top_ratio = expand_stagewise(args.top_ratio, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), name='top_ratio')

    # build vae & var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
        class_token_scales=tuple(args.cls_scales),
    )

    # load checkpoints
    if not osp.exists(args.vae_ckpt):
        os.system(f'wget {HF_HOME}/{osp.basename(args.vae_ckpt)} -O {args.vae_ckpt}')
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    var_sd = torch.load(args.var_ckpt, map_location='cpu')
    if 'trainer' in var_sd: var_sd = var_sd['trainer']['var_wo_ddp']   # allow ar-ckpt-*.pth from train.py
    missing, unexpected = var.load_state_dict(var_sd, strict=False)
    unexpected = [k for k in unexpected if k not in ('spatial_idx', 'attn_bias_cls', 'cls_positions')]   # now non-persistent buffers
    assert not unexpected, f'unexpected keys in {args.var_ckpt}: {unexpected}'
    # class_head is only used by the training loss; checkpoints fine-tuned before it existed still sample fine
    assert all('class_head' in k for k in missing), f'missing keys in {args.var_ckpt}: {missing}'
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'[HierAmp] models ready on {device}; amplify={not args.no_amplify}, '
          f'beta={args.beta}, rho={args.top_ratio}, scales={args.scales}, layers={args.layers}')

    with open(args.mapping_file) as f:
        index_to_wnid = {item['index']: item['wnid'] for item in json.load(f).values()}

    # seeding
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    class_indices = args.classes if args.classes is not None else range(args.start, args.end)
    for ci in class_indices:
        wnid = index_to_wnid[ci]
        save_path = osp.join(args.save_dir, wnid)
        os.makedirs(save_path, exist_ok=True)

        n_saved = 0
        for chunk, offset in enumerate(range(0, args.ipc, args.batch_size)):
            B = min(args.batch_size, args.ipc - offset)
            label_B = torch.full((B,), ci, dtype=torch.long, device=device)
            with torch.inference_mode(), torch.autocast('cuda', enabled=device == 'cuda', dtype=torch.float16, cache_enabled=True):
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=B, label_B=label_B, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                    g_seed=args.seed + chunk,   # distinct seed per chunk so chunks don't repeat
                    amplify=not args.no_amplify, amplify_layers=tuple(args.layers), amplify_scales=tuple(args.scales),
                    beta=beta, top_ratio=top_ratio,
                )
            for k in range(B):
                img = recon_B3HW[k].permute(1, 2, 0).mul(255).cpu().numpy()
                img = PImage.fromarray(img.astype(np.uint8))
                if args.save_reso != img.size[0]:
                    img = img.resize((args.save_reso, args.save_reso), PImage.LANCZOS)
                img.save(osp.join(save_path, f'{n_saved}.png'))
                n_saved += 1
        print(f'[HierAmp] class {ci} ({wnid}): saved {n_saved} images to {save_path}')


if __name__ == '__main__':
    main()
