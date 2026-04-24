"""Download the four foundation-model checkpoints for Phase 3+.

Handled automatically
---------------------
  * scGPT whole-human (Google Drive folder)  -> save/scGPT_human/
  * Geneformer V2-104M (HuggingFace)          -> save/Geneformer/Geneformer-V2-104M/
  * UCE 33-layer + auxiliary files (Figshare) -> save/UCE/

Handled by human (requires SharePoint click-through)
----------------------------------------------------
  * scFoundation 01B-resolution (SharePoint)  -> save/scFoundation/models/

See HUMAN_TASKS.md #3 for the scFoundation manual step.

After all four are on disk, `python -m scripts.download_foundation_models --hash`
re-computes SHA-256 for every checkpoint and writes data/checkpoint_hashes.txt.

Usage
-----
    uv run python scripts/download_foundation_models.py            # download all auto ones
    uv run python scripts/download_foundation_models.py --only geneformer
    uv run python scripts/download_foundation_models.py --hash     # hash-only
    uv run python scripts/download_foundation_models.py --dry-run

Notes
-----
* Licenses (verified 2026-04-24): scGPT MIT, Geneformer Apache-2.0, scFoundation
  Apache-2.0 code + separate Model License for weights, UCE MIT.
* scFoundation weights are under a separate Model License — verify before any
  redistribution (e.g. putting them in a Docker image).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAVE_ROOT = Path("save")
HASH_FILE = Path("data/checkpoint_hashes.txt")

# UCE Figshare record 24320806 -- 33-layer weights + auxiliary files.
UCE_FILES = {
    "33l_8ep_1024t_1280.torch": "https://figshare.com/ndownloader/files/43423236",
    "all_tokens.torch":          "https://figshare.com/ndownloader/files/42706558",
    "species_chrom.csv":         "https://figshare.com/ndownloader/files/42706555",
    "species_offsets.pkl":       "https://figshare.com/ndownloader/files/42706552",
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# scGPT via gdown
# ---------------------------------------------------------------------------
def download_scgpt(dry_run: bool = False) -> Path:
    dest = SAVE_ROOT / "scGPT_human"
    dest.mkdir(parents=True, exist_ok=True)
    folder_id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    log.info(f"[scGPT] gdown --folder {folder_id} -> {dest}")
    if dry_run:
        return dest
    # gdown is installed on-demand so we don't add it to pyproject until needed.
    cmd = [
        sys.executable, "-m", "gdown", "--folder",
        folder_id, "-O", str(dest), "--fuzzy", "--quiet",
    ]
    subprocess.run(cmd, check=True)
    log.info(f"[scGPT] downloaded to {dest}; files: {[p.name for p in dest.iterdir()]}")
    return dest


# ---------------------------------------------------------------------------
# Geneformer via huggingface_hub
# ---------------------------------------------------------------------------
def download_geneformer(dry_run: bool = False, variant: str = "Geneformer-V2-104M") -> Path:
    dest = SAVE_ROOT / "Geneformer"
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"[Geneformer] HuggingFace snapshot_download subfolder={variant} -> {dest}")
    if dry_run:
        return dest
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="ctheodoris/Geneformer",
        allow_patterns=[f"{variant}/*"],
        local_dir=str(dest),
    )
    log.info(f"[Geneformer] downloaded to {path}")
    return Path(path)


# ---------------------------------------------------------------------------
# UCE via curl (Figshare direct, needs browser User-Agent)
# ---------------------------------------------------------------------------
def download_uce(dry_run: bool = False) -> Path:
    dest = SAVE_ROOT / "UCE"
    dest.mkdir(parents=True, exist_ok=True)
    for filename, url in UCE_FILES.items():
        out = dest / filename
        if out.exists() and out.stat().st_size > 0:
            log.info(f"[UCE] skip existing {out} ({out.stat().st_size / 1024**2:.1f} MB)")
            continue
        log.info(f"[UCE] curl {url} -> {out}")
        if dry_run:
            continue
        cmd = [
            "curl", "-L", "-C", "-", "-A", "Mozilla/5.0",
            "--retry", "5", "--retry-delay", "10",
            "--speed-limit", "50000", "--speed-time", "60",
            "-o", str(out), url,
        ]
        subprocess.run(cmd, check=True)
    return dest


# ---------------------------------------------------------------------------
# scFoundation — manual
# ---------------------------------------------------------------------------
def download_scfoundation(dry_run: bool = False) -> Path:
    dest = SAVE_ROOT / "scFoundation" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    msg = (
        "[scFoundation] MANUAL DOWNLOAD REQUIRED\n"
        "  1. Open https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/"
        "IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk in a browser.\n"
        "  2. Choose 'Continue as guest' when prompted (email + consent).\n"
        "  3. Download `models.ckpt` (or `01B-resolution.ckpt`), ~700 MB.\n"
        f"  4. Save to: {dest}/\n"
        "  5. Re-run this script with --hash to register SHA-256.\n"
        "  See HUMAN_TASKS.md #3 for the task-tracking entry."
    )
    log.warning(msg)
    return dest


# ---------------------------------------------------------------------------
# Hash ledger
# ---------------------------------------------------------------------------
def update_hash_ledger() -> None:
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for model_dir in sorted(SAVE_ROOT.glob("*")):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.rglob("*")):
            if not f.is_file():
                continue
            size = f.stat().st_size
            if size < 1024 * 1024:  # skip small config / json
                continue
            digest = _sha256(f)
            rel = f.relative_to(SAVE_ROOT.parent) if SAVE_ROOT.parent in f.parents else f
            entries.append(f"{digest}  {size}  {rel}")
            log.info(f"sha256 {digest[:12]}...  {rel}")
    HASH_FILE.write_text("\n".join(entries) + "\n", encoding="utf-8")
    log.info(f"wrote {HASH_FILE} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODELS = {
    "scgpt":         download_scgpt,
    "geneformer":    download_geneformer,
    "uce":           download_uce,
    "scfoundation":  download_scfoundation,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=list(MODELS.keys()), nargs="+", default=None,
                    help="Download only a subset of models (default: all).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hash", action="store_true",
                    help="Skip downloads; only recompute SHA-256 ledger for existing files.")
    args = ap.parse_args()

    if args.hash:
        update_hash_ledger()
        return

    targets = args.only or list(MODELS.keys())
    log.info(f"target models: {targets}")

    for name in targets:
        fn = MODELS[name]
        try:
            fn(dry_run=args.dry_run)
        except Exception as e:
            log.error(f"[{name}] failed: {e}")
            log.exception(e)

    if not args.dry_run:
        update_hash_ledger()


if __name__ == "__main__":
    main()
