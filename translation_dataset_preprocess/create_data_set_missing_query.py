import os
import pandas as pd
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




def create_ds(raw_target_directory,target_directory ,queries):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    for file in os.listdir(raw_target_directory):
        if file not in queries:
            continue
        rows = {}
        with open(raw_target_directory+file) as qfile:
            for row,line in enumerate(qfile):
                rows[row]={}
                target = line.split("\t")[1].rstrip()
                query =line.split("\t")[0]
                inp = target
                for q in query.split():
                    inp = inp.replace(q,"")+" "+q
                rows[row]["query"]=query
                rows[row]["input_sentence"]=inp
                rows[row]["target_sentence"]=target
            pd.DataFrame.from_dict(rows,orient="index").to_csv(target_directory+file+".csv")

if __name__=="__main__":
    queries_histogram = "query_appearance_histogram.txt"
    queries = read_queries(queries_histogram)
    create_ds("reduced_target/","missing_query_ds/",queries)


