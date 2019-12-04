from summarization.seo_experiment.utils import run_summarization_model,load_file
from nltk import sent_tokenize
def create_files(texts):
    with open("docs.txt",'w') as docs:
        with open("summarization_input.txt",'w') as input:
            for doc in texts:
                docs.write(doc+"\n")
                sentences = sent_tokenize(texts[doc])
                modified_text = " ".join(["<t> " + bytes(s, 'cp1252', "ignore").decode('utf-8', 'ignore').replace("\n"," ").replace("\r", " ") + " </t>" for s in sentences])
                input.write(modified_text+"\n")


texts = load_file("../data/documents.trectext")
create_files(texts)
summarization_models = {"lstm":"summarizations_models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt","transformer":"summarization_models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt"}
summary_kwargs = {"lstm":{"min_length" :"10","block_ngram_repeat": "2"},"transformer":{"min_length" :"3"}}
sum_model ="transformer"
summary_model = summarization_models[sum_model]
run_summarization_model("~/OpenNMT-py/translate.py",summary_model,"summarization_input.txt","competition_doc_summaries",**summary_kwargs[sum_model])