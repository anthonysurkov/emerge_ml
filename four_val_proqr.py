import pandas as pd
import torch
import numpy as np
from two_subopt import Subopts_Graph, Hairpin_Graphs
from two_subopt import sample_subopts_dotbracket, pack_H_mats
from seq_info import GuideInfo, TargetInfo, FlankInfo

NUM_SUBOPTS = 64
EDIT_IN_TARGET = 472
GUIDE_CORE_LEN = 19

df_proqr = (
    pd.read_csv('data/proqr.csv', index_col=0)
      .rename(columns={'proqr_edit': 'edit'})
)
df_proqr['5to3'] = df_proqr['5to3'].str.replace('Z','C')

f_mecp2_r270x = open('data/ref/mecp2_r270x.txt')
mecp2_r270x = f_mecp2_r270x.read().strip()

df_proqr['val_seq'] = df_proqr['5to3'] + '&' + mecp2_r270x

f_mecp2_r270x.close()

def compute_validation_anchors(seq: str, target: str):
    if len(seq) < GUIDE_CORE_LEN:
        core_l_in_guide = 0
        core_r_in_guide = len(seq) - 1
    else:
        pad = (len(seq) - GUIDE_CORE_LEN) // 2
        core_l_in_guide = pad
        core_r_in_guide = pad + GUIDE_CORE_LEN - 1

    eit = int(EDIT_IN_TARGET)

    if not (0 <= eit < len(target)):
        raise ValueError(f"EDIT_IN_TARGET={eit} out of range for target len={len(target)}")

    base = target[eit].upper().replace("T", "U")
    if base != "A":
        raise ValueError(
            f"target[{eit}]='{target[eit]}' not 'A'. "
            f"If this is 1-based, try EDIT_IN_TARGET={eit-1}."
        )

    edit_pos = len(seq) + eit
    guide_l = 0
    guide_r = len(seq) - 1

    return int(edit_pos), int(guide_l), int(guide_r)

def main():
    print('a')
    guides = df_proqr['5to3']
    seqs = df_proqr["val_seq"].tolist()
    editing_rates = df_proqr["edit"].to_numpy(dtype=float)

    # anchors aligned with seqs
    edit_pos, guide_l, guide_r = [], [], []
    for seq in guides:
        ep, gl, gr = compute_validation_anchors(seq, mecp2_r270x)
        edit_pos.append(ep)
        guide_l.append(gl)
        guide_r.append(gr)
    print('a')

    anchors = {
        "edit_pos": np.asarray(edit_pos, dtype=np.int64),
        "guide_l": np.asarray(guide_l, dtype=np.int64),
        "guide_r": np.asarray(guide_r, dtype=np.int64),
    }

    print('a')
    H_mats = []
    for i, hp in enumerate(seqs):
        print(f"{i}: {hp[:50]}")
        dbs = sample_subopts_dotbracket(hp, NUM_SUBOPTS)
        graphs = [Subopts_Graph(hp, db) for db in dbs]
        H_mats.append(Hairpin_Graphs(graphs))

    packed = pack_H_mats(H_mats, y=editing_rates, anchors=anchors)
    torch.save(packed, "data/proqr_val_subopts.pt")

if __name__ == "__main__":
    main()
