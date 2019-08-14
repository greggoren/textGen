import os

def make_new_file(old_file,new_file):
    with open(new_file,"w") as nfile:
        with open(old_file) as ofile:
            for line in ofile:
                input_sequence = line.split(",")[2]
                query = line.split(",")[1]
                new_input_sequence = input_sequence+" "+query
                new_row = ",".join([line.split(",")[0],line.split(",")[1],new_input_sequence,line.split(",")[3]])
                nfile.write(new_row)

def prepare_query_based_ds(original_dir,new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in os.listdir(original_dir):
        make_new_file(original_dir+file,new_dir+file)


prepare_query_based_ds("translations_new_ds/","translations_new_ds_with_query/")

