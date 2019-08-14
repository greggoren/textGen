import os
import pandas as pd

def number_of_unique_targets(ds_dir):
    unique_num_file = open("unique_target_exploration_stats.txt","w")
    for file in os.listdir(ds_dir):
        df = pd.read_csv(ds_dir+file,header=0,delimiter=",")
        count_target_values = df.target_sentence.nunique()
        unique_num_file.write(file+" "+str(count_target_values)+"\n")
    unique_num_file.close()


def variance_of_target_appearance(ds_dir):
    variance_file = open("variance_target_exploration_stats.txt","w")
    for file in os.listdir(ds_dir):
        df = pd.read_csv(ds_dir+file,header=0,delimiter=",")
        variance_target_values = df.target_sentence.value_counts().var()
        variance_file.write(file+" "+str(variance_target_values)+"\n")
    variance_file.close()



number_of_unique_targets('translations_new_ds_with_query/')
variance_of_target_appearance('translations_new_ds_with_query/')