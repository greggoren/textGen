from summarization.seo_experiment.utils import run_summarization_model
from summarization.seo_experiment.summarization_process import chosen_sentence_for_replacement
from nltk import sent_tokenize
# run_summarization_model("script","model.bat","inp","outp",min_length = '1',max_tokenization = "10")
import javaobj

with open('ROUND-08-195-51','rb') as fd:

    pobj = javaobj.load(fd)

#1.5762436503169919
print(float(pobj['made']))
