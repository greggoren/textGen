src = open("input.txt","w")
trgt = open("input.txt","w")
with open("") as file:
    for line in file:
        s = line.split(",")[2]
        t = line.split(",")[3].rstrip()
        src.write(s+"\n")
        trgt.write(t+"\n")
    src.close()
    trgt.close()
