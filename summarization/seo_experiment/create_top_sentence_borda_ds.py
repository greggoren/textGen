from nltk import sent_tokenize
from summarization.seo_experiment.evaluation.analysis import read_trec_file
from summarization.seo_experiment.utils import load_file
from summarization.seo_experiment.workingset_creator import read_queries_file
import os


def read_summary_data(fname):
    chosen_sentences = {}
    chosen_indexes = {}
    with open(fname) as file:
        for i,line in enumerate(file):
            if i==0:
                continue
            doc = line.split("\t")[1]
            index = line.split("\t")[2]
            sentence = line.split("\t")[3]
            chosen_sentences[doc]=sentence
            chosen_indexes[doc]=index
    return chosen_indexes,chosen_sentences



def create_files(ranked_lists,ref_index,top_docs_index,queries,output_dir,chosen_indexes,texts,chosen_sentences):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+"queries.txt",'w') as queries_file:
        with open(output_dir+"source.txt",'w') as source_file:
            with open(output_dir+"sentence_pool.txt",'w') as target_sentences:
                with open(output_dir+"all_input.txt",'w') as all_input_file:
                    all_input_file.write("query\tdocname\tsentence_index\tsentence\ttarget_sentence\n")
                    for epoch in ranked_lists:
                        for query in ranked_lists[epoch]:
                            ref_doc = ranked_lists[epoch][query][ref_index]
                            top_docs = ranked_lists[epoch][query][:top_docs_index]
                            fixed_query = str(int(query))+str(epoch).zfill(2)
                            chosen_index = chosen_indexes[ref_doc]
                            chosen_sentence = chosen_sentences[ref_doc]
                            for doc in top_docs:
                                text = texts[doc]
                                sentences = sent_tokenize(text)
                                for sentence in sentences:
                                    source_file.write(chosen_sentence+"\n")
                                    queries_file.write(queries[fixed_query]+"\n")
                                    target_sentences.write(sentence)
                                    all_input_file.write(fixed_query+"\t"+ref_doc+"\t"+chosen_index+"\t"+chosen_sentence.rstrip()+"\t"+sentence.rstrip().replace("\n","")+"\n")


if __name__=="__main__":
    trec_file = "trecs/trec_file_original_sorted.txt"
    queries_file = "data/queries_seo_exp.xml"
    queries = read_queries_file(queries_file)
    ranked_lists = read_trec_file(trec_file)
    texts = load_file("data/documents.trectext")
    chosen_indexes,chosen_sentences = read_summary_data("input_data/all_data_transformer.txt")
    create_files(ranked_lists,-1,3,queries,"top_sentences_borda/",chosen_indexes,texts,chosen_sentences)