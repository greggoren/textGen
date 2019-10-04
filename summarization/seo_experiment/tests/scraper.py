from summarization.seo_experiment.utils import load_file
import nltk
import numpy as np

texts = load_file("../data/documents.trectext")
stats={}
for doc in texts:
    r = doc.split("-")[1]
    if r not in ["06","07"]:
        continue
    if r not in stats:
        stats[r]=[]
    stats[r].append(len(nltk.sent_tokenize(texts[doc])))

for r in stats:
    stats[r]=np.mean(stats[r])

print(stats)



