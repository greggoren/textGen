from summarization.seo_experiment.utils import run_summarization_model
from summarization.seo_experiment.summarization_process import chosen_sentence_for_replacement

# run_summarization_model("script","model.bat","inp","outp",min_length = '1',max_tokenization = "10")

sentences = ["a shit is crazy", "crazy is my name yes yes yes yes yes yes yes yes yes yes yes yes","yes yes yes yes yes yes yes yes yes yes yes"]
query = "crazy shit"
print(chosen_sentence_for_replacement(sentences,query))
