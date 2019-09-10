from optparse import OptionParser
import os
from summarization.seo_experiment.utils import load_file

def get_working_items(texts,rounded=True):
    workingset = {}
    if rounded:
        for docid in texts:
            epoch = docid.split("-")[1]
            if int(epoch)==0:
                continue
            query = docid.split("-")[2]
            updated_query = str(int(query))+epoch
            if updated_query not in workingset:
                workingset[updated_query]=[]
            workingset[updated_query].append(docid)
    else:
        pass
    return workingset

def create_workingset(workingset,fname):
    with open(fname,'w') as f:
        for query in workingset:
            for i,docid in enumerate(workingset[query]):
                f.write(query + ' Q0 ' + docid + ' ' + str(i+1) + ' -' + str(i+1) + ' indri\n')


if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("--rounded", dest="rounded")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--out_file", dest="out_file")
    (options, args) = parser.parse_args()

    doc_texts = load_file(options.trectext_file)
    if options.rounded.lower()=="true":
        workingset = get_working_items(doc_texts,True)
    if not os.path.exists(os.path.dirname(options.out_file)):
        os.makedirs(os.path.dirname(options.out_file))
    create_workingset(workingset,options.out_file)