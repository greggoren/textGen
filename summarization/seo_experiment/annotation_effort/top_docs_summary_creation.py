from summarization.seo_experiment.utils import load_file,read_trec_file,run_summarization_model
from summarization.seo_experiment.borda_mechanism import read_queries
import nltk

def create_summarization_ds(ranked_lists,texts):
    with open("summarization_data.txt",'w',encoding="utf-8") as sum_data:
        with open("texts_for_summary.txt","w",encoding="utf-8") as text_data:
            for r in range(1,7):
                epoch = str(r).zfill(2)
                for query in ranked_lists[epoch]:
                    winner = ranked_lists[epoch][query][0]
                    text = texts[winner]
                    sentences = nltk.sent_tokenize(text)
                    line = " ".join(["<t> "+s.replace("\n","")+" </t>" for s in sentences])+"\n"
                    text_data.write(line)
                    sum_data.write(epoch+"\t"+query+"\t"+winner+"\n")

if __name__=="__main__":
    trectext_file = "../data/documents.trectext"
    # queries_file = "../data/queries.txt"
    trec_file = "../trecs/trec_file_original_sorted.txt"
    summary_kwargs = {"lstm":{"min_length" :"10","block_ngram_repeat": "2"},"transformer":{"min_length" :"3"}}
    ranked_lists = read_trec_file(trec_file)
    texts = load_file(trectext_file)
    create_summarization_ds(ranked_lists,texts)
    run_summarization_model("~/OpenNMT-py/translate.py","../summarization_models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt","texts_for_summary.txt","top_docs_summaries.txt",**summary_kwargs["transformer"])