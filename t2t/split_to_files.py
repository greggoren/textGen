q = []
with open("example/queries_file") as file:
    for line in file:
        q.append(line.rstrip())
acc = 0
c =0
with open("input_encoded.out") as file:
    for i,line in enumerate(file):
        if q[i] in line:
            acc+=1
        c+=1
    print(acc/c)