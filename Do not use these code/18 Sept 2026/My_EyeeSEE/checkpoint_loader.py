"""
Phase 3 - Universal Checkpoint Loader
Supports: .pth / .pt (PyTorch) and .safetensors formats
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple
import torch
from huggingface_hub import hf_hub_download, list_repo_files


def detect_checkpoint_format(checkpoint_path: str) -> str:
    """
    Detect format of a single checkpoint file.
    Returns: 'safetensors', 'pytorch', or 'unknown'
    """
    p = Path(checkpoint_path)
    if p.suffix == '.safetensors':
        return 'safetensors'
    if p.suffix in ('.pth', '.pt'):
        return 'pytorch'
    return 'unknown'


def load_state_dict_from_file(file_path: str, device: torch.device) -> dict:
    """
    Load a state dict from either .safetensors or .pth/.pt file.
    Returns the raw state dict.
    """
    fmt = detect_checkpoint_format(file_path)

    if fmt == 'safetensors':
        try:
            import safetensors.torch
            return safetensors.torch.load_file(file_path, device=str(device))
        except ImportError:
            raise ImportError("Install safetensors: pip install safetensors")

    if fmt == 'pytorch':
        return torch.load(file_path, map_location=device, weights_only=False)

    raise ValueError(f"Unrecognised file format: {file_path}")


def load_model_weights(model, checkpoint_path: str, device: torch.device) -> None:
    """
    Load only model weights from a single .pth/.pt or .safetensors file.
    Handles both raw state-dicts and wrapped checkpoints produced by train.py.
    """
    raw = load_state_dict_from_file(checkpoint_path, device)

    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    else:
        state_dict = raw

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


def load_full_checkpoint_dir(
    model,
    trainer,
    checkpoint_dir: str,
    device: torch.device
) -> int:
    """
    Load a full training checkpoint from a directory.

    Directory layout (new SafeTensors format from train_fixed.py):
        checkpoint_dir/
            model.safetensors
            optimizer.safetensors
            scheduler.safetensors
            metadata.json

    Also handles legacy single-file .pth layout:
        checkpoint_epoch_N.pth  (contains model + optimizer + scheduler)

    Returns:
        next_epoch (int): epoch to resume from
    """
    checkpoint_dir = Path(checkpoint_dir)

    # ── New multi-file SafeTensors layout ──
    metadata_path = checkpoint_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        for key, attr in [
            ('model',     ('model', None)),
            ('optimizer', ('optimizer', trainer.optimizer)),
            ('scheduler', ('scheduler', trainer.scheduler)),
        ]:
            for ext in ('safetensors', 'pt'):
                fp = checkpoint_dir / f'{key}.{ext}'
                if fp.exists():
                    state = load_state_dict_from_file(str(fp), device)
                    if key == 'model':
                        model.load_state_dict(state)
                        model.to(device)
                    else:
                        attr[1].load_state_dict(state)
                    break

        trainer.best_val_loss = metadata.get('best_val_loss', float('inf'))
        last_epoch = metadata['epoch']
        next_epoch = last_epoch + 1

        print(f"Checkpoint loaded  (epoch {last_epoch})")
        print(f"  Val Loss : {metadata.get('metrics', {}).get('loss', 'N/A')}")
        print(f"  Disc Dice: {metadata.get('metrics', {}).get('disc_dice', 'N/A')}")
        print(f"  Cup Dice : {metadata.get('metrics', {}).get('cup_dice', 'N/A')}")
        print(f"Resuming from epoch {next_epoch}")
        return next_epoch

    # ── Legacy single-file .pth layout ──
    pth_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if pth_files:
        pth_file = sorted(pth_files)[-1]
        ckpt = torch.load(str(pth_file), map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

        if 'optimizer_state_dict' in ckpt:
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        trainer.best_val_loss = ckpt.get(
            'best_val_loss', ckpt.get('metrics', {}).get('loss', float('inf'))
        )

        last_epoch = ckpt['epoch']
        next_epoch = last_epoch + 1
        print(f"Legacy checkpoint loaded  (epoch {last_epoch})")
        print(f"Resuming from epoch {next_epoch}")
        return next_epoch

    raise FileNotFoundError(f"No valid checkpoint found in: {checkpoint_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_hf_checkpoints(repo_id: str, token: Optional[str] = None) -> list:
    """
    Return all checkpoint folder names in Nj-1111/EyeeSEE/checkpoints/
    sorted by epoch number (ascending).

    Works for both:
        checkpoints/epoch_050/        (new format)
        checkpoints/checkpoint_epoch_50.pth  (legacy format)
    """
    import re
    token = token or os.getenv('HF_TOKEN_2') or os.getenv('HF_TOKEN')

    files = list_repo_files(repo_id=repo_id, token=token)

    epoch_map = {}

    for f in files:
        # New directory format
        m = re.search(r'checkpoints/epoch_(\d+)/', f)
        if m:
            ep = int(m.group(1))
            folder = f'checkpoints/epoch_{ep:03d}'
            epoch_map[ep] = folder
            continue

        # Legacy single-file format
        m = re.search(r'checkpoints/checkpoint_epoch_(\d+)\.pth', f)
        if m:
            ep = int(m.group(1))
            epoch_map[ep] = f

    return [(ep, path) for ep, path in sorted(epoch_map.items())]


def download_checkpoint_for_inference(
    repo_id: str,
    epoch: Optional[int] = None,
    token: Optional[str] = None,
    local_dir: str = '/kaggle/working/ckpt_inference'
) -> Tuple[str, str]:
    """
    Download a checkpoint from HuggingFace for inference.

    Args:
        repo_id: HF repo ID
        epoch:   Specific epoch to download.  None → downloads latest.
        token:   HF token
        local_dir: Where to save files

    Returns:
        (local_path, fmt)  where fmt is 'safetensors_dir' or 'pytorch_file'
    """
    token = token or os.getenv('HF_TOKEN_2') or os.getenv('HF_TOKEN')
    checkpoints = list_hf_checkpoints(repo_id, token)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {repo_id}")

    if epoch is None:
        ep, ckpt_ref = checkpoints[-1]
    else:
        matched = [(e, p) for e, p in checkpoints if e == epoch]
        if not matched:
            raise ValueError(f"Epoch {epoch} not found. Available: {[e for e, _ in checkpoints]}")
        ep, ckpt_ref = matched[0]

    local_dir = Path(local_dir) / f'epoch_{ep:03d}'
    local_dir.mkdir(parents=True, exist_ok=True)

    # SafeTensors directory format
    if ckpt_ref.endswith('/') or not ckpt_ref.endswith('.pth'):
        all_files = list_repo_files(repo_id=repo_id, token=token)
        ckpt_files = [f for f in all_files if f.startswith(ckpt_ref)]

        for hf_filename in ckpt_files:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                token=token,
                local_dir=str(local_dir)
            )

        print(f"Downloaded checkpoint (epoch {ep}) → {local_dir}")
        return str(local_dir), 'safetensors_dir'

    # Legacy single .pth file
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_ref,
        token=token,
        local_dir=str(local_dir)
    )
    print(f"Downloaded checkpoint (epoch {ep}) → {local_path}")
    return local_path, 'pytorch_file'


def load_model_for_inference(model, repo_id, epoch=None, device=None, token=None):
    """
    Download checkpoint from HF and load weights into model.
    Handles .pt and .pth, detects epoch automatically.
    """
    import os, re
    from huggingface_hub import hf_hub_download, list_repo_files

    token  = token  or os.getenv('HF_TOKEN_2') or os.getenv('HF_TOKEN')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files  = list_repo_files(repo_id=repo_id, token=token)
    epochs = {}
    for f in files:
        m = re.search(r'checkpoints/checkpoint_epoch_(\d+)\.pt(?:h?)$', f)
        if m:
            epochs[int(m.group(1))] = f

    if not epochs:
        raise FileNotFoundError(f"No checkpoints found in {repo_id}")

    target_epoch = epoch if epoch is not None else max(epochs)
    if target_epoch not in epochs:
        raise ValueError(f"Epoch {target_epoch} not found. Available: {sorted(epochs)}")

    ckpt_file = hf_hub_download(
        repo_id=repo_id,
        filename=epochs[target_epoch],
        token=token
    )

    ckpt       = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from epoch {target_epoch} on {device}")
    return model
