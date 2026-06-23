import torch
import numpy as np
from two_subopt import Subopts_Graph, Hairpin_Graphs
from two_subopt import sample_subopts_dotbracket, pack_H_mats
from seq_info import GuideInfo, TargetInfo, FlankInfo

data = {
    "ggac": [63.4, 58.9, 47.4],
    "ac":   [20.9, 13.0, 7.82],
    "nd1":  [0.2, 0.8, 2.4],
    "nd2":  [3.33, 4.78, 0.69],
    "nd3":  [8.71, 9.41, 10.1],
    "nd4":  [5.92, 5.48, 7.84],
    "nd5":  [4.79, 6.8],
    "nd6":  [14.6, 9.88, 1.54],
    "nd7":  [12.9, 28.3, 10.3],
    "nd9":  [35.1, 48.7, 48.9],
    "nd10": [6, 8.9, 46.5, 16.2, 10.9, 6.2],
    "nd11": [9.6, 13.1, 7],
    "nd12": [26.9, 28, 30.5],
    "nd13": [10.8, 12.3, 15.3],
    "nd14": [41.7, 23.5, 15, 1.9, 5.2],
}
means = {k: float(np.mean(v)) / 100 for k, v in data.items()}

NUM_SUBOPTS = 64
EDIT_IN_TARGET = 95          # 0-based index INSIDE target_seq (you said target A is 95)
GUIDE_CORE_LEN = 12          # the “middle 12 nt” you care about

def get_validation_parts(guide_id: str):
    guide_info = GuideInfo(guide_id)
    target_id = guide_info.target
    target_info = TargetInfo(target_id)

    just_guide_seq = guide_info.guide_seq
    flanks = FlankInfo(target_id)
    guide_seq = flanks.left + just_guide_seq + flanks.right

    target_seq = target_info.v_seq
    return guide_seq, just_guide_seq, flanks, target_seq

def get_validation_sequence(guide_id: str) -> str:
    guide_seq, _, _, target_seq = get_validation_parts(guide_id)
    print(f"{guide_seq}&{target_seq}")
    return f"{guide_seq}&{target_seq}"

def compute_validation_anchors(guide_id: str):
    guide_seq, just_guide_seq, flanks, target_seq = get_validation_parts(guide_id)

    # --- guide core (middle 12 nt of just_guide_seq, but don't crash if shorter)
    if len(just_guide_seq) < GUIDE_CORE_LEN:
        core_l_in_guide = 0
        core_r_in_guide = len(just_guide_seq) - 1
    else:
        pad = (len(just_guide_seq) - GUIDE_CORE_LEN) // 2
        core_l_in_guide = pad
        core_r_in_guide = pad + GUIDE_CORE_LEN - 1

    guide_l = len(flanks.left) + core_l_in_guide
    guide_r = len(flanks.left) + core_r_in_guide

    # --- edited A location inside target_seq
    eit = int(EDIT_IN_TARGET)

    if not (0 <= eit < len(target_seq)):
        raise ValueError(f"{guide_id}: EDIT_IN_TARGET={eit} out of range for target_seq len={len(target_seq)}")

    base = target_seq[eit].upper().replace("T", "U")
    if base != "A":
        raise ValueError(
            f"{guide_id}: target_seq[{eit}]='{target_seq[eit]}' not 'A'. "
            f"If this is 1-based, try EDIT_IN_TARGET={eit-1}."
        )

    # --- IMPORTANT: convert to compact coords: guide_seq then target_seq (after removing '&')
    edit_pos = len(guide_seq) + eit

    # sanity
    if not (0 <= guide_l <= guide_r < len(guide_seq)):
        raise ValueError(
            f"{guide_id}: guide bounds out of range: guide_l={guide_l}, guide_r={guide_r}, guide_seq len={len(guide_seq)}"
        )

    return int(edit_pos), int(guide_l), int(guide_r)

def main():
    guide_ids = list(data.keys())
    seqs = [get_validation_sequence(gid) for gid in guide_ids]
    editing_rates = [means[gid] for gid in guide_ids]

    # anchors aligned with seqs
    edit_pos, guide_l, guide_r = [], [], []
    for gid in guide_ids:
        ep, gl, gr = compute_validation_anchors(gid)
        edit_pos.append(ep); guide_l.append(gl); guide_r.append(gr)

    anchors = {
        "edit_pos": np.asarray(edit_pos, dtype=np.int64),
        "guide_l":  np.asarray(guide_l, dtype=np.int64),
        "guide_r":  np.asarray(guide_r, dtype=np.int64),
    }

    H_mats = []
    for i, hp in enumerate(seqs):
        print(f"{i}, {hp}")
        dbs = sample_subopts_dotbracket(hp, NUM_SUBOPTS)
        graphs = [Subopts_Graph(hp, db) for db in dbs]
        H_mats.append(Hairpin_Graphs(graphs))


    if gid == guide_ids[0]:
        guide_seq, just_guide_seq, flanks, target_seq = get_validation_parts(gid)
        print("[debug] lens:",
              "guide_seq", len(guide_seq),
              "just_guide_seq", len(just_guide_seq),
              "flank_left", len(flanks.left),
              "flank_right", len(flanks.right),
              "target_seq", len(target_seq))
        print("[debug] anchors:",
              "EDIT_IN_TARGET", EDIT_IN_TARGET,
              "edit_pos(compact)", ep,
              "guide_l", gl,
              "guide_r", gr,
              "target_base", target_seq[EDIT_IN_TARGET])

    packed = pack_H_mats(H_mats, y=editing_rates, anchors=anchors)
    torch.save(packed, "data/r255x_val_subopts.pt")

if __name__ == "__main__":
    main()
