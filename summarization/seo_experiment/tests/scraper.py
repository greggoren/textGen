from summarization.seo_experiment.utils import run_summarization_model
from summarization.seo_experiment.summarization_process import chosen_sentence_for_replacement
from nltk import sent_tokenize
# run_summarization_model("script","model.bat","inp","outp",min_length = '1',max_tokenization = "10")

# sentences = ["a shit is crazy", "crazy is my name yes yes yes yes yes yes yes yes yes yes yes yes","yes yes yes yes yes yes yes yes yes yes yes"]
# query = "crazy shit"
# print(chosen_sentence_for_replacement(sentences,query))
paragraph = "vines shade	the implementation of mechanical harvesting is often stimulated by changes in labor laws, labor shortages, and bureaucratic complications. it can be expensive to hire labor for short periods of time, which does not square well with the need to reduce production costs and harvest quickly, often at night. however, very small vineyards, incompatible widths between rows of grape vines and steep terrain hinder the employment of machine harvesting even more than the resistance of traditional views which reject such harvesting."
sentences = sent_tokenize(paragraph)
paragraph = " ".join(["<t> " + s + " </t>" for s in sentences])
print(paragraph)