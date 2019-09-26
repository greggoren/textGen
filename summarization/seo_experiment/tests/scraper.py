import pandas as pd
d1 = {"a":[1,2,1,2],"d":[3,2,1,2],"c":[1,1,1,2]}


df = pd.DataFrame.from_dict(d1)
df = df[df.a>1]
print(df)
for i,row in df.iterrows():
    print(i,row)
    print(df.ix[i])




