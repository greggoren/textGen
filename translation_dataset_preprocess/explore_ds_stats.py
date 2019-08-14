import os
import pandas as pd
import random
import math
from t2t.utils import run_bash_command

def number_of_unique_targets(ds_dir):
    unique_num_file = open("unique_target_exploration_stats.txt","w")
    for file in os.listdir(ds_dir):
        df = pd.read_csv(ds_dir+file,header=0,delimiter=",")
        count_target_values = df.target_sentence.nunique()
        unique_num_file.write(file+" "+str(count_target_values)+"\n")
    unique_num_file.close()


def create_train_test(ds_dir):
    valid_queries = []
    for file in os.listdir(ds_dir):
        df = pd.read_csv(ds_dir+file,header=0,delimiter=",")
        count_target_values = df.target_sentence.nunique()
        std_target_values = df.target_sentence.value_counts().std()
        if count_target_values>=1000 and std_target_values<=50:
            valid_queries.append(file)
    random.seed(9001)
    train_index = math.floor(0.9*len(valid_queries))
    random.shuffle(valid_queries)
    train_folder = ds_dir.replace("/","")+"_train/"
    test_folder = ds_dir.replace("/","")+"_test/"
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for file in valid_queries[:train_index]:
        run_bash_command("cp "+ds_dir+file+ " "+train_folder+file)
    for file in valid_queries[train_index:]:
        run_bash_command("cp " + ds_dir + file + " " + test_folder + file)



def variance_of_target_appearance(ds_dir):
    variance_file = open("variance_target_exploration_stats.txt","w")
    for file in os.listdir(ds_dir):
        df = pd.read_csv(ds_dir+file,header=0,delimiter=",")
        variance_target_values = df.target_sentence.value_counts().var()
        variance_file.write(file+" "+str(variance_target_values)+"\n")
    variance_file.close()



# number_of_unique_targets('translations_new_ds_with_query/')
# variance_of_target_appearance('translations_new_ds_with_query/')
create_train_test('translations_new_ds_with_query/')