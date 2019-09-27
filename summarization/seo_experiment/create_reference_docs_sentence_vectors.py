import os
import logging
from optparse import OptionParser
import sys
from summarization.seo_experiment.utils import load_file
import nltk
from gen_utils import run_bash_command

def reference_docs_calculation(stats,ref_index):
    return {q:stats[q][ref_index] for q in stats}

def get_reference_docs(trec_file, index):
    stats = read_trec_file(trec_file)
    return reference_docs_calculation(stats,index)

def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[2]
            if query not in stats:
                stats[query]=[]
            stats[query].append(doc)
    return stats


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("--trec_file", dest="trec_file")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--ref_index", dest="ref_index")
    parser.add_option("--index", dest="index")
    parser.add_option("--sentences_out_file", dest="sentences_out_file")
    parser.add_option("--vectors_output_dir", dest="vectors_output_dir")
    (options, args) = parser.parse_args()
    if not os.path.exists(options.vectors_output_dir):
        os.makedirs(options.vectors_output_dir)
    if not os.path.exists(os.path.dirname(options.sentences_out_file)):
        os.makedirs(os.path.dirname(options.sentences_out_file))
    reference_docs = get_reference_docs(options.trec_file,int(options.ref_index))
    document_text = load_file(options.trectext_file)
    with open(options.sentences_out_file,'w') as out_file:
        for query in reference_docs:
            doc = reference_docs[query]
            doc_text = document_text[doc]
            sentences = nltk.sent_tokenize(doc_text)
            for i,sentence in enumerate(sentences):
                out_file.write(query+"\t"+doc+"_"+str(i)+"\t"+sentence.rstrip().replace("\n","")+"\n")

    command = " ~/jdk1.8.0_181/bin/java -Djava.library.path=/lv_local/home/sgregory/indri-5.6/swig/obj/java/ -cp seo_summarization.jar PrepareTFIDFVectorsReferenceDocs "+options.index+" "+options.sentences_out_file+" "+options.vectors_output_dir
    logger.info("## Running vector creation command: "+command+" ##")
    logger.info(run_bash_command(command))
    logger.info("Vector creation is DONE..")





