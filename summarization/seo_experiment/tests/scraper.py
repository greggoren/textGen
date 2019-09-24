from summarization.seo_experiment.utils import run_summarization_model
from summarization.seo_experiment.summarization_process import chosen_sentence_for_replacement
from nltk import sent_tokenize
# run_summarization_model("script","model.bat","inp","outp",min_length = '1',max_tokenization = "10")
import javaobj
from summarization.seo_experiment.borda_mechanism import dict_cosine_similarity,cosine_similarity
d1 = {"a":1,"d":2,"c":1}
d2 = {"a":1,"b":2,"c":1}
v1 = [1,0,1,2]
v2 = [1,2,1,0]

print(dict_cosine_similarity(d1,d2))
print(cosine_similarity(v1,v2))



