from random import shuffle
from math import floor
import os
def read_queries(fname):
    queries =[]
    with open(fname) as file:
        for line in file:
            query = line.split(":")[0]
            amount = int(line.split(":")[1].rstrip())
            if amount < 10000:
                continue

            queries.append("_".join(query.split()))
    return queries



def split_sets(train,data_dir,train_dir,test_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    for fname in os.listdir(data_dir):
        if fname in train:
            os.popen("cp "+data_dir+fname+" "+train_dir+fname)
        else:
            os.popen("cp " + data_dir + fname + " " + test_dir + fname)

if __name__=="__main__":
    queries_histogram = "query_appearance_histogram.txt"
    queries = read_queries(queries_histogram)
    shuffle(queries)
    train_index = floor(0.9*len(queries))
    train = queries[:train_index]
    data_dir = "missing_query_ds/"

