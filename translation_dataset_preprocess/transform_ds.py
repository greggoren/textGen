import os
input_dir = "new_input_pull/"
new_input_dir = "input_dir/"
if not os.path.exists(new_input_dir):
    os.makedirs(new_input_dir)
for file in os.listdir(input_dir):
    new_file_name = new_input_dir+file
    flag = True
    with open(input_dir+file) as original_file:
        with open(new_file_name,'w') as new_file:
            for line in original_file:
                if flag:
                    flag=False
                    continue
                new_line = "\t".join(line.split(",")[1:]).rstrip()+"\n"
                new_file.write(new_line)