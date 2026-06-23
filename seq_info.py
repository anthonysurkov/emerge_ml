import pandas as pd
from dataclasses import dataclass

ref_dir = 'data/ref'

@dataclass
class IndexRange:
    left: int
    right: int

    def __iter__(self):
        yield self.left
        yield self.right

class TargetInfo:
    def __init__(
        self,
        target_id: str,
        ref_json: str = f'{ref_dir}/targets.json'
    ):
        df = pd.read_json(ref_json)
        df = df[df['target_id'] == target_id]

        self.target_id = target_id
        self.m_seq = df['m_seq'].item()
        self.v_seq = df['v_seq'].item()
        self.target_seq = df['target_seq'].item()

        # 0-idx:
        m_target_idxs = df['m_target_idx'].item()
        m_target_idxs = m_target_idxs.split(':')
        self.m_target_idxs = IndexRange(
            left = int(m_target_idxs[0]),
            right = int(m_target_idxs[1])
        )
        v_target_idxs = df['v_target_idx'].item()
        v_target_idxs = v_target_idxs.split(':')
        self.v_target_idxs = IndexRange(
            left = int(v_target_idxs[0]),
            right = int(v_target_idxs[1])
        )

        m_guide_idxs = df['m_guide_idx'].item()
        m_guide_idxs = m_guide_idxs.split(':')
        self.m_guide_idxs = IndexRange(
            left = int(m_guide_idxs[0]),
            right = int(m_guide_idxs[1])
        )

        self.m_At_idx = int(df['m_At_idx'].item())
        self.v_At_idx = int(df['v_At_idx'].item())

class GuideInfo:
    def __init__(
        self,
        guide_id: str,
        ref_json: str = f'{ref_dir}/guides.json'
    ):
        df = pd.read_json(ref_json)
        df = df[df['guide_id'] == guide_id]

        self.guide_id = guide_id
        self.guide_seq = df['seq'].item()
        self.guide_legacy = df['legacy'].item()
        self.target = df['target'].item()

class FlankInfo:
    def __init__(
        self,
        target_id: str,
        ref_json: str = f'{ref_dir}/flanks.json'
    ):
        df = pd.read_json(ref_json)
        df = df[df['target_id'] == target_id]
        self.target_id = target_id

        left = df[df['flank_id'] == 'left']
        self.left = left['seq'].item()

        right = df[df['flank_id'] == 'right']
        self.right = right['seq'].item()
