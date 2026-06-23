import pandas as pd

TARGET_ID = 'r270x_z'
NUM_SEQS = 65536

EDIT_CUTOFF = 0
RANDOM_STATE = 123

DATA_SRC = "data"
TRGT_SRC = f"{DATA_SRC}/ref/targets.json"

INFILE = f"{DATA_SRC}/emerge/{TARGET_ID}.csv"
OUTFILE = f"{DATA_SRC}/{TARGET_ID}_{NUM_SEQS}.csv"

EDIT_POS = 30 # (all 0-idx)
GUIDE_L = 54 # start of n10 sequence
GUIDE_R = 63 # end of n10 sequence

# Load hairpin information
trgt = pd.read_json(TRGT_SRC)
HAIRPIN_SEQ = (trgt.loc[trgt['target_id'] == TARGET_ID, 'm_seq'].squeeze())
N10_REG = 'NNNNZNNNN'

# Load data
df = pd.read_csv(INFILE)
df['hairpin'] = df['5to3'].map(lambda x: HAIRPIN_SEQ.replace(N10_REG, x))
df['5to3'] = df['5to3'].str.upper().str.replace('T','U')
df['hairpin'] = df['hairpin'].str.upper().str.replace('T','U')
# Select training data
df = df[df['n'] >= 10].head(NUM_SEQS)
#df_high = df[df['mle'] >= EDIT_CUTOFF]
#df_low  = df[df['mle'] <  EDIT_CUTOFF].sort_values(by='mle', ascending=False).head(5000)
#half = NUM_SEQS // 2
#df_high = df_high.sample(n=half, random_state=RANDOM_STATE)
#df_low  = df_low.sample(n=half, random_state=RANDOM_STATE)
#df = pd.concat([df_high, df_low]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# TODO: fxn to load edit pos, guide_l, guide_r positions
df['edit_pos'] = EDIT_POS
df['guide_l'] = GUIDE_L
df['guide_r'] = GUIDE_R

edit_pos = HAIRPIN_SEQ[EDIT_POS]
n10_reg = HAIRPIN_SEQ[GUIDE_L:GUIDE_R]
print(HAIRPIN_SEQ)
print(edit_pos)
print(n10_reg)

print(df)
df.to_csv(OUTFILE, index=False)
