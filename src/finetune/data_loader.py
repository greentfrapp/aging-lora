"""AnnData -> Geneformer rank-value tokenized batches.

Geneformer rank-value encoding (V2/104M):
    1. For each cell, normalize raw counts: x_norm = x / sum(x) * 1e4
    2. Divide by gene_median (per-gene non-zero median across Genecorpus 104M)
    3. Sort genes by descending normalized-by-median value
    4. Take top-(seq_len-1) genes; prepend <cls> token
    5. Pad with <pad> to seq_len

We avoid materializing the full count matrix; cells are streamed from a
backed AnnData via __getitem__ on demand.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


GENEFORMER_DIR = Path("save/Geneformer/geneformer")


@dataclass
class GeneformerVocab:
    token_dict: dict
    gene_median: dict
    pad_token_id: int
    cls_token_id: int

    @classmethod
    def load(cls, gf_dir: Path = GENEFORMER_DIR) -> "GeneformerVocab":
        with open(gf_dir / "token_dictionary_gc104M.pkl", "rb") as f:
            token_dict = pickle.load(f)
        with open(gf_dir / "gene_median_dictionary_gc104M.pkl", "rb") as f:
            gene_median = pickle.load(f)
        return cls(
            token_dict=token_dict,
            gene_median={k: float(v) for k, v in gene_median.items()},
            pad_token_id=token_dict["<pad>"],
            cls_token_id=token_dict["<cls>"],
        )


def build_var_token_arrays(var_df, vocab: GeneformerVocab):
    """Map AnnData var rows -> (gene_token_id, gene_median) per column.

    Genes with no Ensembl ID, no token, or no median are flagged with token_id=-1.
    """
    n_genes = len(var_df)
    token_ids = np.full(n_genes, -1, dtype=np.int64)
    medians = np.full(n_genes, np.nan, dtype=np.float32)
    ensembl_ids = var_df["ensembl_id"].astype(object).fillna("").to_numpy()
    for i, eid in enumerate(ensembl_ids):
        if not eid:
            continue
        tid = vocab.token_dict.get(eid)
        med = vocab.gene_median.get(eid)
        if tid is None or med is None or med <= 0:
            continue
        token_ids[i] = tid
        medians[i] = med
    return token_ids, medians


class TokenizedAnnData(Dataset):
    """Backed-AnnData dataset that yields (input_ids, attention_mask, age) per cell.

    Cells are referenced by integer index into ``self.indices``, which is an
    array of row positions in the underlying ``.h5ad`` (post-filter).
    """

    def __init__(
        self,
        h5ad_path: Path,
        indices: np.ndarray,
        vocab: GeneformerVocab,
        seq_len: int = 2048,
        ages: np.ndarray | None = None,
        donors: np.ndarray | None = None,
    ):
        self.h5ad_path = Path(h5ad_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.vocab = vocab
        self.seq_len = seq_len
        self._adata: ad.AnnData | None = None
        self._token_ids: np.ndarray | None = None
        self._medians: np.ndarray | None = None
        self._valid_cols: np.ndarray | None = None  # boolean mask of usable genes
        self.ages = ages
        self.donors = donors

    def _ensure_open(self):
        if self._adata is None:
            self._adata = ad.read_h5ad(self.h5ad_path, backed="r")
            tids, meds = build_var_token_arrays(self._adata.var, self.vocab)
            valid = tids >= 0
            self._token_ids = tids[valid]
            self._medians = meds[valid]
            self._valid_cols = np.where(valid)[0]

    def __len__(self):
        return len(self.indices)

    def _tokenize_cell(self, counts_row: np.ndarray) -> np.ndarray:
        # counts_row: dense float array over self._valid_cols subset
        total = counts_row.sum()
        if total <= 0:
            ids = np.full(self.seq_len, self.vocab.pad_token_id, dtype=np.int64)
            ids[0] = self.vocab.cls_token_id
            return ids
        norm = counts_row * (1e4 / total) / self._medians
        nz = norm > 0
        if not nz.any():
            ids = np.full(self.seq_len, self.vocab.pad_token_id, dtype=np.int64)
            ids[0] = self.vocab.cls_token_id
            return ids
        # rank descending by norm value among non-zero genes
        nz_idx = np.where(nz)[0]
        order = nz_idx[np.argsort(-norm[nz_idx], kind="stable")]
        top = order[: self.seq_len - 1]
        token_ids = np.full(self.seq_len, self.vocab.pad_token_id, dtype=np.int64)
        token_ids[0] = self.vocab.cls_token_id
        token_ids[1 : 1 + len(top)] = self._token_ids[top]
        return token_ids

    def __getitem__(self, i):
        self._ensure_open()
        row_pos = int(self.indices[i])
        X = self._adata.X
        # X is a backed CSR; row slicing returns a sparse matrix
        row = X[row_pos : row_pos + 1]
        if sp.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()
        sub = row[self._valid_cols].astype(np.float32, copy=False)
        ids = self._tokenize_cell(sub)
        attn = (ids != self.vocab.pad_token_id).astype(np.int64)
        # Geneformer: <cls> is also attended; it's not a pad-id, so attn==1 there.
        out = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(attn),
        }
        if self.ages is not None:
            out["age"] = torch.tensor(float(self.ages[i]), dtype=torch.float32)
        if self.donors is not None:
            out["donor"] = str(self.donors[i])
        return out


def _stringify(arr) -> np.ndarray:
    return np.asarray(arr, dtype=object).astype(str)


def select_indices(
    h5ad_path: Path,
    cell_type: str,
    cohorts: Sequence[str] | None = None,
    exclude_donors: Sequence[str] | None = None,
    include_donors: Sequence[str] | None = None,
    max_cells_per_donor: int | None = None,
    rng_seed: int = 0,
):
    """Resolve a cell-type/cohort/donor filter to row indices + ages + donor labels.

    Returns (indices, ages, donor_ids).
    """
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs
    mask = (obs["cell_type"].astype(str) == cell_type).to_numpy()
    if cohorts is not None:
        mask &= np.isin(_stringify(obs["cohort_id"]), list(cohorts))
    donor_ids = _stringify(obs["donor_id"])
    if exclude_donors is not None:
        mask &= ~np.isin(donor_ids, list(exclude_donors))
    if include_donors is not None:
        mask &= np.isin(donor_ids, list(include_donors))
    idx = np.where(mask)[0]
    ages = obs["age"].to_numpy()[idx].astype(np.float32)
    donors = donor_ids[idx]
    a.file.close()

    if max_cells_per_donor is not None:
        rng = np.random.default_rng(rng_seed)
        keep = []
        keep_ages = []
        keep_donors = []
        for donor in np.unique(donors):
            mask_d = donors == donor
            sub = idx[mask_d]
            sub_ages = ages[mask_d]
            if len(sub) > max_cells_per_donor:
                sel = rng.choice(len(sub), size=max_cells_per_donor, replace=False)
                sub = sub[sel]
                sub_ages = sub_ages[sel]
            keep.append(sub)
            keep_ages.append(sub_ages)
            keep_donors.append(np.full(len(sub), donor, dtype=object))
        idx = np.concatenate(keep)
        ages = np.concatenate(keep_ages).astype(np.float32)
        donors = np.concatenate(keep_donors)
    return idx, ages, donors
