from summarization.seo_experiment.utils import load_file

data = load_file("data/documents.trectext")
queries = set()
for doc in data:
    query = doc.split("-")[2]
    queries.add(query)

with open("data/queries.txt") as f1:
    with open("data/queries_comp.txt",'w') as f2:
        for line in f1:
            if line.split(":")[0] in queries:
                f2.write(line)