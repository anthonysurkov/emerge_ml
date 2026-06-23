import pandas as pd

scaffold0_flank5 = "GGCACCACACUCCCCGU" # X.0 scaffold
scaffold0_flank3 = "CCCGUUU"
scaffold1_flank5 = "CCCG"              # X.1 scaffold
scaffold1_flank3 = "CCCGUU"
scaffold2_flank5 = "CCCGU"             # X.2 scaffold
scaffold2_flank3 = "CCCGUU"

df = pd.read_csv("r270x_z_val.csv")
df = df.dropna()
df['5to3'] = df['5to3'].str.upper()

def get_flank(flank_id: int, seq_5to3: str) -> str:
    if flank_id == 0:
        return scaffold0_flank5 + seq_5to3 + scaffold0_flank3
    elif flank_id == 1:
        return scaffold1_flank5 + seq_5to3 + scaffold1_flank3
    else:
        return scaffold2_flank5 + seq_5to3 + scaffold2_flank3


df['eon'] = df.apply(lambda row: get_flank(flank_id=row['flank_designation'], seq_5to3=row['5to3']), axis=1)

df.to_csv("r270x_z_eons.csv", index=False)

with open("actin_beta_target.fasta", "r") as f:
    text = f.read()
print(text[1575])
