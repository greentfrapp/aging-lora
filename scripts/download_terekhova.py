"""Download the Terekhova 2023 Immunity PBMC atlas from Synapse.

Target file
-----------
By default, downloads `syn56693935` (`raw_counts_h5ad.tar.gz`, 9.8 GB) — a single
tar containing the raw-counts h5ad for all 166 donors. This is the minimum viable
artefact for our pipeline:
  * Foundation models require raw counts (not log-normalized), and this file ships
    raw counts directly in h5ad format — no R→Python conversion, no re-counting.
  * It is substantially smaller than the full project root (syn49637038 ≈ 1811
    files, raw Cell Ranger outputs including TCR/BCR, >100 GB).

The project is organized as:
    syn49637038 (Project root)
    ├── BCR_Processed       — B-cell receptor outputs (not needed)
    ├── Cytek               — flow cytometry (not needed)
    ├── Demultiplexing      — HTO demux outputs (not needed; already applied)
    ├── GEX_HTO_processed   — scRNA-seq processed data  (syn50542388)
    │   ├── all_pbmcs.tar.gz         15.9 GB  — full Seurat atlas
    │   ├── raw_counts_h5ad.tar.gz    9.8 GB  — <-- DEFAULT TARGET (syn56693935)
    │   ├── {cell_type}.tar.gz              — per-cell-type Seurat subsets
    │   └── ...
    ├── RAW                 — raw Cell Ranger (~100+ GB, not needed)
    └── TCR_Processed       — T-cell receptor outputs (not needed)

Prerequisites
-------------
* A free Synapse account.
* The data use conditions on syn49637038 accepted in the Synapse UI.
* Authentication via one of:
    - ~/.synapseConfig under [authentication] with  authtoken = <PAT>   (canonical)
    - ~/.synapseConfig with the raw PAT on line 1                       (tolerated fallback)
    - env-variable SYNAPSE_AUTH_TOKEN

Usage
-----
    uv run python scripts/download_terekhova.py                     # default: raw_counts_h5ad.tar.gz
    uv run python scripts/download_terekhova.py --dry-run
    uv run python scripts/download_terekhova.py --synapse-id syn51197006   # full Seurat atlas instead
    uv run python scripts/download_terekhova.py --synapse-id syn49637038   # whole project (>100 GB!)

Notes
-----
* syncFromSynapse is checksum-aware — re-running the script after a partial
  transfer only re-downloads the missing/incomplete files.
* After download, extract:  tar -xzf data/cohorts/raw/terekhova/raw_counts_h5ad.tar.gz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("terekhova-download")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _read_authtoken() -> str | None:
    """Resolve a Synapse PAT from env or ~/.synapseConfig.

    Supports two ~/.synapseConfig formats:
      * Canonical INI:      [authentication]\n authtoken = <PAT>
      * Raw single-line:    just the PAT on line 1 (tolerant fallback)
    """
    import os

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token:
        return token

    cfg = Path.home() / ".synapseConfig"
    if not cfg.exists():
        return None

    text = cfg.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    # Canonical INI: delegate to synapseclient via login() default behavior.
    if text.lstrip().startswith("["):
        return None  # let synapseclient read the file itself

    # Raw token on a single line
    first = text.splitlines()[0].strip()
    if first and not first.startswith("#"):
        return first
    return None


def _login():
    try:
        import synapseclient
    except ImportError as e:
        raise SystemExit(
            "synapseclient is not installed. Run: uv sync  (and make sure "
            "synapseclient is in pyproject.toml dependencies)."
        ) from e

    authtoken = _read_authtoken()
    if authtoken:
        log.info("authenticating with an explicit PAT (from env or raw ~/.synapseConfig)")
        # Pass a non-existent configPath so synapseclient's ConfigParser does not
        # try to parse the user's ~/.synapseConfig — which errors out if the
        # file is a raw token rather than proper INI format. The Synapse source
        # comments: "Does not fail if the file does not exist".
        bogus_cfg = Path.home() / ".synapseConfig.nonexistent"
        syn = synapseclient.Synapse(skip_checks=True, configPath=str(bogus_cfg))
        syn.login(authToken=authtoken, silent=False)
    else:
        # Canonical INI ~/.synapseConfig path — let synapseclient read it.
        syn = synapseclient.login(silent=False)
    return syn, synapseclient


def _describe(syn, entity_id: str):
    entity = syn.get(entity_id, downloadFile=False)
    concrete = getattr(entity, "concreteType", "") or ""
    name = getattr(entity, "name", "<unnamed>")
    log.info(f"{entity_id} -> {name}  ({concrete.split('.')[-1]})")
    return entity, concrete


def _list_files(syn, entity_id: str) -> list[dict]:
    """Recurse via syn.getChildren and collect all File-concreteType descendants."""
    out: list[dict] = []
    stack = [entity_id]
    while stack:
        parent = stack.pop()
        for child in syn.getChildren(parent):
            ctype = child.get("type", "") or child.get("concreteType", "")
            if "File" in ctype:
                out.append(child)
            elif "Folder" in ctype or "Project" in ctype:
                stack.append(child["id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synapse-id", default="syn56693935",
                    help="Synapse entity to download (default: syn56693935 = raw_counts_h5ad.tar.gz, 9.8 GB). "
                         "Use syn51197006 for the full Seurat atlas (15.9 GB), or syn49637038 for the whole "
                         "project (>100 GB — includes raw Cell Ranger, TCR, BCR, etc.).")
    ap.add_argument("--out-dir", default="data/cohorts/raw/terekhova",
                    help="Destination directory (default: data/cohorts/raw/terekhova).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log the entity tree and sizes without downloading.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _configure_logging(args.verbose)

    syn, synapseclient = _login()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity, concrete = _describe(syn, args.synapse_id)

    # Enumerate files first so we can show a plan before we start downloading.
    if "File" in concrete:
        children = [{"id": args.synapse_id, "name": entity.name, "type": concrete}]
    else:
        log.info(f"enumerating files under {args.synapse_id} ...")
        children = _list_files(syn, args.synapse_id)

    if not children:
        log.error("no File entities resolved under %s — nothing to download", args.synapse_id)
        return 2

    log.info(f"{len(children)} file entities to download -> {out_dir}")
    for c in children[:50]:
        log.info(f"  {c['id']}  {c.get('name', '?')}")
    if len(children) > 50:
        log.info(f"  ... and {len(children) - 50} more")

    if args.dry_run:
        log.info("--dry-run set; exiting before transfer")
        return 0

    # syncFromSynapse handles recursive tree sync, resume, and checksum skip.
    log.info("starting syncFromSynapse — may take a while for large files")
    try:
        synapseutils = synapseclient.synapseutils
    except AttributeError:
        # newer synapseclient versions expose utils as a separate module
        import synapseutils as synapseutils  # type: ignore
    files = synapseutils.syncFromSynapse(syn, args.synapse_id, path=str(out_dir))
    log.info(f"download complete: {len(files)} file(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
