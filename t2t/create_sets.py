import os
data_dir = "data/"
os.popen("mkdir example")
os.popen("mkdir train")
for file in os.listdir(data_dir):
    filename = data_dir+file
    os.popen("tail -10 "+filename+" >> example/validation.csv")
    os.popen("head -9991 "+filename+ " >> train/train_set.csv")
