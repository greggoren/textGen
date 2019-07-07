import os


reduced_input_dir = "reduced_input/"
reduced_target_dir = "reduced_target/"

if os.path.exists(reduced_input_dir):
    os.makedirs(reduced_input_dir)
if os.path.exists(reduced_target_dir):
    os.makedirs(reduced_target_dir)

input_dir = "data/"
target_dir = "translations_pool/"

def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result

queries = read_queries("queries.txt")

for q in queries:
    inp_command = "head -10000 "+input_dir+q+" > "+reduced_input_dir+q
    tar_command = "head -10000 "+target_dir+q+" > "+reduced_target_dir+q
    os.popen(inp_command)
    os.popen(tar_command)
