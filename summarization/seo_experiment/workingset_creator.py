from optparse import OptionParser
import os
from summarization.seo_experiment.utils import load_file



def read_queries_file(queries_file):
    last_number_state = None
    stats = {}
    with open(queries_file) as file:
        for line in file:
            if "<number>" in line:
                last_number_state = line.replace('<number>','').replace('</number>',"").split("_")[0].rstrip()
            if '<text>' in line:
                stats[last_number_state]=line.replace('<text>','').replace('</text>','').rstrip()
    return stats

def modify_queries_file(texts,queries_stats,queries_out_file):
    if not os.path.exists(os.path.dirname(queries_out_file)):
        os.makedirs(os.path.dirname(queries_out_file))
    with open(queries_out_file,'w') as out_file:
        out_file.write('<parameters>\n')
        for docid in texts:
            epoch = docid.split("-")[1]
            if int(epoch)==0:
                continue
            query = docid.split("-")[2]
            updated_query = str(int(query))+epoch
            query_text = queries_stats[query]
            out_file.write("<query>\n")
            out_file.write("<number>"+updated_query+"</number>\n")
            out_file.write("<text>"+query_text+"</text>\n")
            out_file.write("</query>\n")
        out_file.write("</parameters>\n")





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
    parser.add_option("--mode", dest="mode")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--out_file", dest="out_file")
    parser.add_option("--queries_file", dest="queries_file")
    parser.add_option("--queries_out_file", dest="queries_out_file")
    (options, args) = parser.parse_args()
    doc_texts = load_file(options.trectext_file)
    if options.mode == "workingset":
        if options.rounded.lower()=="true":
            workingset = get_working_items(doc_texts,True)
        if not os.path.exists(os.path.dirname(options.out_file)):
            os.makedirs(os.path.dirname(options.out_file))
        create_workingset(workingset,options.out_file)
    elif options.mode=="queries":
        queries_stats = read_queries_file(options.queries_file)
        modify_queries_file(doc_texts,queries_stats,options.queries_out_file)