# import in pandas to allow for reading
import pandas as pd

# make pandas read in the data provided
df1 = pd.read_csv("all_tro.members.txt")
df1.to_csv("all_tro.members.csv", index=None)
print("tro_syn.txt -> tro_syn.csv done")

df2 = pd.read_csv("tro.syn.txt")
df2.to_csv("tro.syn.csv", index=None)
print("all_tro_members.txt -> tro_members.csv done")
