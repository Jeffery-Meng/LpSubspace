import sys
import os
import pandas as pd
import glob

# common files to be summarized
common = sys.argv[1]
output = sys.argv[2]

if len(sys.argv) > 3:
    print(len(sys.argv))
    exit(1)
    
df = None
file_list = glob.glob(common)
for file_path in file_list:
    cur_df = pd.read_csv(file_path)
    if df is None:
        df = cur_df
    else:
        df = pd.concat([df, cur_df])
df.to_csv(output)

