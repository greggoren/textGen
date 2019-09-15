from summarization.seo_experiment.utils import run_summarization_model
from summarization.seo_experiment.summarization_process import chosen_sentence_for_replacement
from nltk import sent_tokenize
# run_summarization_model("script","model.bat","inp","outp",min_length = '1',max_tokenization = "10")
import javaobj

def write_files(**kwargs):
    for key,val in kwargs.items():
        val[0].write(val[1]+"\n")


with open("test1.txt",'w') as t1:
    with open("test2.txt",'w') as t2:
        args = {"t1":(t1,"hi"),"t2":(t2,"bye")}
        write_files(**args)




