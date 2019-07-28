import os
input_dir = "new_input_pull/"
new_input_dir = "input_dir/"

for file in os.listdir(input_dir):
    new_file_name = new_input_dir+file
    with open(input_dir+file) as original_file:
        with open(new_file_name,'w') as new_file:
            for line in original_file:
                new_line = "\t".join(line.split(",")[1:]).rstrip()+"\n"
                new_file.write(new_line)