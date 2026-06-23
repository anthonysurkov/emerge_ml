import RNA
import pandas as pd
import numpy as np
import torch
import os

from dataclasses import dataclass
from typing import List

BASE2ID = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
NUM_SUBOPTS = 256  # K
NUM_SEQS = 65536
TARGET_ID = 'r270x_z'

DATA_SRC = 'data'
INFILE = f'{DATA_SRC}/{TARGET_ID}_{NUM_SEQS}.csv'
OUTFILE = f'{DATA_SRC}/training/{TARGET_ID}_subopts_{NUM_SEQS}_k{NUM_SUBOPTS}.pt'

CHUNK_SIZE = 2048  # tune down if still OOM; 1024 is safe on ~32GB

class Subopts_Graph:
    def __init__(self, seq: str, db: str):
        self.seq = seq
        self.db = db
        self.nodes, self.node_pos, self.back_edges, self.pair_edges = self._dotbracket2graph(seq, db)
        self.n = self.nodes.shape[0]

    @staticmethod
    def _dotbracket2graph(seq: str, db: str):
        seq = seq.strip().upper().replace('T', 'U')
        db = db.strip()

        strand_breaks = [i for i, ch in enumerate(db) if ch == '&']
        db_nosep = db.replace('&', '')
        seq_nosep = seq.replace('&', '')
        n = len(seq_nosep)
        assert len(db_nosep) == n

        # node_pos: compact coordinate (after removing '&')
        node_pos = np.arange(n, dtype=np.int64)

        # 4-base onehot + paired flag
        x = np.zeros((n, 5), dtype=np.float32)
        for i, b in enumerate(seq_nosep):
            x[i, BASE2ID[b]] = 1.0

        # base-pair edges
        pt = RNA.ptable(db_nosep)
        pair_edges = []
        for i in range(1, n + 1):
            j = pt[i]
            if j > i:
                pair_edges.append((i - 1, j - 1))
                pair_edges.append((j - 1, i - 1))
                x[i - 1, 4] = 1.0
                x[j - 1, 4] = 1.0

        # backbone edges sans connections across '&'
        cut = set(strand_breaks)

        back_edges = []
        compact_i = -1
        orig_to_compact = {}
        for orig_i, ch in enumerate(db):
            if ch == "&":
                continue
            compact_i += 1
            orig_to_compact[orig_i] = compact_i

        for orig_i in range(len(db) - 1):
            if orig_i in cut or (orig_i + 1) in cut:
                continue
            if db[orig_i] == "&" or db[orig_i + 1] == "&":
                continue
            a = orig_to_compact[orig_i]
            b = orig_to_compact[orig_i + 1]
            back_edges.append((a, b))
            back_edges.append((b, a))

        return (
            x,
            node_pos,
            np.array(back_edges, dtype=np.int64),
            np.array(pair_edges, dtype=np.int64),
        )


@dataclass
class Hairpin_Graphs:
    graphs: List[Subopts_Graph]

    @property
    def n(self) -> int:
        return len(self.graphs)


def normalize_seq(seq: str) -> str:
    return str(seq).strip().upper().replace("T", "U")


def sample_subopts_dotbracket(seq: str, k: int) -> List[str]:
    md = RNA.md()
    md.uniq_ML = 1

    fc = RNA.fold_compound(seq, md)
    mfe_ss, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()

    out = []
    for s in fc.pbacktrack(k, RNA.PBACKTRACK_NON_REDUNDANT):
        if isinstance(s, bytes):
            s = s.decode()
        out.append(str(s))
    return out


def pack_H_mats(H_mats, y=None, n_obs=None, k_obs=None, anchors=None):
    """
    anchors (optional): dict with keys:
      - "edit_pos":  (num_seqs,) int64
      - "guide_l":   (num_seqs,) int64
      - "guide_r":   (num_seqs,) int64
    All anchors should be in COMPACT coordinates (after removing '&').
    """

    X_list = []
    POS_list = []
    EI_list = []
    ET_list = []  # edge_type: 0 backbone, 1 pair

    graph_ptr = [0]
    edge_ptr = [0]
    seq_ptr = [0]

    total_nodes = 0
    total_edges = 0
    total_graphs = 0

    for seq_obj in H_mats:
        graphs = seq_obj.graphs
        for g in graphs:
            x = g.nodes.astype(np.float32)
            n = x.shape[0]

            be = g.back_edges.astype(np.int64)
            pe = g.pair_edges.astype(np.int64)

            if be.size:
                be_shift = be + total_nodes
                EI_list.append(be_shift)
                ET_list.append(np.zeros((be_shift.shape[0],), dtype=np.int64))
            if pe.size:
                pe_shift = pe + total_nodes
                EI_list.append(pe_shift)
                ET_list.append(np.ones((pe_shift.shape[0],), dtype=np.int64))

            X_list.append(x)
            POS_list.append(g.node_pos.astype(np.int64))

            total_nodes += n
            graph_ptr.append(total_nodes)

            e_added = (be.shape[0] if be.size else 0) + (pe.shape[0] if pe.size else 0)
            total_edges += e_added
            edge_ptr.append(total_edges)

            total_graphs += 1

        seq_ptr.append(total_graphs)

    X = torch.from_numpy(np.vstack(X_list))  # (total_nodes, F)

    if EI_list:
        edge_index = torch.from_numpy(np.vstack(EI_list).T)    # (2, total_edges)
        edge_type = torch.from_numpy(np.concatenate(ET_list))  # (total_edges,)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    out = {
        "X": X,
        "node_pos": torch.from_numpy(np.concatenate(POS_list)),  # (total_nodes,)
        "edge_index": edge_index,
        "edge_type": edge_type,
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "seq_ptr": torch.tensor(seq_ptr, dtype=torch.long),
    }

    if y is not None:
        out["y"] = torch.tensor(np.asarray(y, dtype=np.float32))
    if n_obs is not None and k_obs is not None:
        out["n"] = torch.tensor(np.asarray(n_obs, dtype=np.int32))
        out["k"] = torch.tensor(np.asarray(k_obs, dtype=np.int32))

    if anchors is not None:
        out["anchors"] = {
            "edit_pos": torch.tensor(np.asarray(anchors["edit_pos"], dtype=np.int64)),
            "guide_l":  torch.tensor(np.asarray(anchors["guide_l"], dtype=np.int64)),
            "guide_r":  torch.tensor(np.asarray(anchors["guide_r"], dtype=np.int64)),
        }

    return out

def _compact_len(seq: str) -> int:
    seq = normalize_seq(seq)
    return len(seq.replace("&", ""))

def _validate_anchors_for_seq(seq: str, edit_pos: int, guide_l: int, guide_r: int):
    L = _compact_len(seq)
    if not (0 <= edit_pos < L):
        raise ValueError(f"edit_pos {edit_pos} out of range for L={L}")
    if not (0 <= guide_l < L) or not (0 <= guide_r < L):
        raise ValueError(f"guide bounds ({guide_l},{guide_r}) out of range for L={L}")
    if guide_l > guide_r:
        raise ValueError(f"guide_l > guide_r: ({guide_l},{guide_r})")

def _orig_to_compact_index(seq: str, idx: int) -> int:
    """
    Convert an index in the ORIGINAL sequence (which may include '&') to compact coords.
    If your CSV is already compact coords, do not use this.
    """
    seq = normalize_seq(seq)
    if idx < 0 or idx >= len(seq):
        raise ValueError(f"orig idx {idx} out of range for seq len {len(seq)}")
    if seq[idx] == "&":
        raise ValueError("index points to '&'")

    # count non-& up to idx
    return sum(1 for ch in seq[:idx] if ch != "&")

def main():
    df = pd.read_csv(INFILE)
    seqs = df["hairpin"].map(normalize_seq).tolist()

    n = df["n"].to_numpy(dtype=np.int32)
    k = df["k"].to_numpy(dtype=np.int32)
    editing_rates = df["mle"].to_numpy(dtype=np.float32)

    CSV_INDICES_ARE_ORIGINAL = False
    edit_pos = df["edit_pos"].to_numpy()
    guide_l = df["guide_l"].to_numpy()
    guide_r = df["guide_r"].to_numpy()

    edit_pos_c = np.empty(len(seqs), dtype=np.int64)
    guide_l_c = np.empty(len(seqs), dtype=np.int64)
    guide_r_c = np.empty(len(seqs), dtype=np.int64)

    for i, hp in enumerate(seqs):
        ep, gl, gr = int(edit_pos[i]), int(guide_l[i]), int(guide_r[i])
        if CSV_INDICES_ARE_ORIGINAL:
            ep = _orig_to_compact_index(hp, ep)
            gl = _orig_to_compact_index(hp, gl)
            gr = _orig_to_compact_index(hp, gr)
        _validate_anchors_for_seq(hp, ep, gl, gr)
        edit_pos_c[i], guide_l_c[i], guide_r_c[i] = ep, gl, gr

    num_chunks = (len(seqs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    outdir = f'{DATA_SRC}/training'
    os.makedirs(outdir, exist_ok=True)

    for chunk_idx in range(num_chunks):
        lo = chunk_idx * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, len(seqs))
        print(f"Chunk {chunk_idx}/{num_chunks}  seqs [{lo}, {hi})")

        H_mats = []
        for i in range(lo, hi):
            hp = seqs[i]
            print(f"  seq {i}: {hp}")
            dbs    = sample_subopts_dotbracket(hp, NUM_SUBOPTS)
            graphs = [Subopts_Graph(hp, db) for db in dbs]
            H_mats.append(Hairpin_Graphs(graphs))

        anchors = {
            "edit_pos": edit_pos_c[lo:hi],
            "guide_l":  guide_l_c[lo:hi],
            "guide_r":  guide_r_c[lo:hi],
        }

        packed = pack_H_mats(
            H_mats,
            y=editing_rates[lo:hi],
            anchors=anchors,
        )

        chunk_path = f'{outdir}/{TARGET_ID}_subopts_{NUM_SEQS}_k{NUM_SUBOPTS}_chunk{chunk_idx:04d}.pt'
        torch.save(packed, chunk_path)
        print(f"  saved → {chunk_path}")

        # explicitly free before next chunk
        del H_mats, packed

if __name__ == "__main__":
    main()
