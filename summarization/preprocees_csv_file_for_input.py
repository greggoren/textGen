import pandas as pd
import sys

def read_df(fname):
    return pd.read_csv(fname,delimiter=",",names=["a","b","query","source","input"])

def prepare_input(df):
    tmp = df["input"]
    tmp.reset_index(drop=True)
    with open("input.txt",'w') as f:
        for row in tmp.itertuples():
            f.write(row[1].replace("\"",""))

if __name__=="__main__":
    fname = sys.argv[1]
    df = read_df(fname)
    prepare_input(df)
