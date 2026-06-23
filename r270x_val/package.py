import torch
import numpy as np
import pandas as pd
import re

from two_subopt import Subopts_Graph, Hairpin_Graphs
from two_subopt import sample_subopts_dotbracket, pack_H_mats

NUM_SUBOPTS = 64
EDIT_IN_TARGET = 1575      # 0-based index inside beta-actin target sequence

EON_CSV = "r270x_z_eons.csv"
TARGET_FASTA = "actin_beta_target.fasta"
OUTFILE = "r270x_z_beta_actin_subopts.pt"


def normalize_rna(seq):
    seq = str(seq).strip().upper().replace("T", "U")

    if not re.fullmatch(r"[ACGUZ]+", seq):
        bad = sorted(set(re.sub(r"[ACGUZ]", "", seq)))
        raise ValueError(f"Invalid characters in sequence: {bad}")

    return seq


def read_target_seq(path):
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            seq_lines.append(line)

    target_seq = normalize_rna("".join(seq_lines))

    if "Z" in target_seq:
        raise ValueError("Target sequence contains Z. Z should only appear in EONs.")

    return target_seq


def read_eons(path):
    eon_df = pd.read_csv(path)

    if "eon" not in eon_df.columns:
        raise ValueError(f"No 'eon' column found in {path}. Columns: {list(eon_df.columns)}")

    if "mean" not in eon_df.columns:
        raise ValueError(f"No 'mean' column found in {path}. Columns: {list(eon_df.columns)}")

    eon_df = eon_df.dropna(subset=["eon", "mean"]).reset_index(drop=True)

    eon_df["eon"] = eon_df["eon"].apply(normalize_rna)
    eon_df["mean"] = eon_df["mean"].astype(float)

    if eon_df["mean"].max() > 1:
        eon_df["mean"] = eon_df["mean"] / 100

    return eon_df


def get_z_window(eon_seq, flank=4):
    matches = list(re.finditer("Z", eon_seq))

    if len(matches) != 1:
        raise ValueError(f"EON must contain exactly one Z, found {len(matches)}: {eon_seq}")

    z_idx = matches[0].start()

    guide_l = z_idx - flank
    guide_r = z_idx + flank

    if guide_l < 0 or guide_r >= len(eon_seq):
        raise ValueError(
            f"Z-centered window out of bounds for EON: {eon_seq}. "
            f"z_idx={z_idx}, guide_l={guide_l}, guide_r={guide_r}"
        )

    variable_region = eon_seq[guide_l:guide_r + 1]

    if not re.fullmatch(r"[ACGU]{4}Z[ACGU]{4}", variable_region):
        raise ValueError(
            f"Bad Z-centered variable region: {variable_region} from {eon_seq}. "
            "Expected exactly 4 nt upstream, Z, and 4 nt downstream."
        )

    return int(guide_l), int(guide_r), variable_region


def get_validation_sequence(eon_seq, target_seq):
    hp = f"{eon_seq}&{target_seq}"
    print(hp)
    return hp


def compute_validation_anchors(eon_seq, target_seq):
    guide_l, guide_r, variable_region = get_z_window(eon_seq, flank=4)

    eit = int(EDIT_IN_TARGET)

    if not (0 <= eit < len(target_seq)):
        raise ValueError(
            f"EDIT_IN_TARGET={eit} out of range for target_seq len={len(target_seq)}"
        )

    base = target_seq[eit]

    if base != "A":
        raise ValueError(
            f"target_seq[{eit}]='{target_seq[eit]}' not 'A'. "
            f"If this is 1-based, try EDIT_IN_TARGET={eit - 1}."
        )

    edit_pos = len(eon_seq) + eit

    return int(edit_pos), int(guide_l), int(guide_r), variable_region


def main():
    eon_df = read_eons(EON_CSV)
    target_seq = read_target_seq(TARGET_FASTA)

    eon_seqs = eon_df["eon"].tolist()
    editing_rates = eon_df["mean"].to_numpy(dtype=float)

    seqs = [get_validation_sequence(eon, target_seq) for eon in eon_seqs]

    edit_pos, guide_l, guide_r, variable_regions = [], [], [], []

    for eon_seq in eon_seqs:
        ep, gl, gr, vr = compute_validation_anchors(eon_seq, target_seq)
        edit_pos.append(ep)
        guide_l.append(gl)
        guide_r.append(gr)
        variable_regions.append(vr)

    anchors = {
        "edit_pos": np.asarray(edit_pos, dtype=np.int64),
        "guide_l": np.asarray(guide_l, dtype=np.int64),
        "guide_r": np.asarray(guide_r, dtype=np.int64),
    }

    print("[debug] n_eons:", len(eon_seqs))
    print("[debug] target_seq len:", len(target_seq))
    print("[debug] target base at edit:", target_seq[EDIT_IN_TARGET])
    print("[debug] first eon:", eon_seqs[0])
    print("[debug] first variable region:", variable_regions[0])
    print("[debug] first anchors:",
          "edit_pos", edit_pos[0],
          "guide_l", guide_l[0],
          "guide_r", guide_r[0])

    seqs_for_folding = [seq.replace("Z", "C") for seq in seqs]
    H_mats = []

    for i, hp in enumerate(seqs_for_folding):
        print(f"{i}, {hp}")
        dbs = sample_subopts_dotbracket(hp, NUM_SUBOPTS)
        graphs = [Subopts_Graph(hp, db) for db in dbs]
        H_mats.append(Hairpin_Graphs(graphs))

    packed = pack_H_mats(
        H_mats,
        y=editing_rates,
        anchors=anchors,
    )

    torch.save(packed, OUTFILE)
    print(f"[saved] {OUTFILE}")


if __name__ == "__main__":
    main()
