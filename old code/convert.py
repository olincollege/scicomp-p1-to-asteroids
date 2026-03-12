import pandas as pd

# read in tro_syn and convert to csv
df1 = pd.read_csv(
    "tro.syn.txt",
    sep=r"\s+",
    comment="%",
    header=None,
    names=[
        "asteroid_number",
        "H",
        "da_AU",
        "D_deg",
        "f_degy",
        "e_p",
        "g_arcsec_y",
        "sin_i_p",
        "s_arcsec_y",
        "L",
        "My",
    ],
)
df1.to_csv("tro_syn.csv", index=None)
print("tro.syn.txt -> tro_syn.csv done")

# read in all_tro.members and convert to csv
df2 = pd.read_csv(
    "all_tro.members.txt",
    sep=r"\s+",
    comment="%",
    header=None,
    names=[
        "asteroid_number",
        "H_mag",
        "status",
        "family1",
        "dv_fam1",
        "near1",
        "family2",
        "dv_fam2",
        "near2",
        "rescod",
    ],
    low_memory=False,
)
df2.to_csv("tro_members.csv", index=None)
print("all_tro.members.txt -> tro_members.csv done")
