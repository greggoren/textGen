q = []
with open("example/queries_file") as file:
    for line in file:
        q.append(line.rstrip())
acc = 0
c =0
with open("input_encoded.out") as file:
    for i,line in enumerate(file):
        splitted = q[i].split()
        for term in splitted:
            if term in line:
                acc+=1
                break
        c+=1
    print(acc/c)