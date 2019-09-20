from gen_utils import run_bash_command


summarization_command = """python summarization_process.py --mode=all --summary_input_file=input_data/input_paragraphs_transformer.txt --sum_model=transformer --trectext_file=data/documents.trectext --trec_file=trecs/trec_file_original_sorted.txt --ref_index=-1 --candidate_dir=~/textGen/summarization/summarization_pool/ --queries_file=data/queries_seo_exp.xml --model_file=/lv_local/home/sgregory/textGen/summarization/wiki.en.bin --summary_script_file=~/OpenNMT-py/translate.py --summary_output_file=summaries/output --number_of_top_docs=3 --sentences_vectors_dir=sentence_ref_vectors/ --documents_vectors_dir=asr_tfidf_vectors/"""

vectors_command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/lv_local/home/sgregory/indri-5.6/swig/obj/java/ -cp seo_summarization.jar PrepareTFIDFVectorsText merged_indices/merged_index/ input_data/all_data_transformer.txt summaries/output_transformer.txt summary_vectors/"

updata_text_command = "python choose_summaries_for_replacement.py --doc_tfidf_dir=asr_tfidf_vectors/ --summaries_tfidf_dir=summary_vectors/ --queries_file=input_data/queries_transformer.txt --summaries_file=summaries/output_transformer.txt --input_data_file=input_data/all_data_transformer.txt --trec_file=trecs/trec_file_original_sorted.txt --number_of_top_docs=3 --trectext_file=data/documents.trectext --new_trectext_file=data/updated_documents.trectext --new_ws_file=data/updated_workingset --model_file=/lv_local/home/sgregory/textGen/summarization/wiki.en.bin"

rerank_command = "python ranking_process.py --mode=all --features_dir=Features_post --merged_index=merged_indices/merged_index_post --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_original_post --workingset_file=data/workingset_original --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_post.txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_post --trectext_file=data/updated_documents.trectext --home_path=~/ --base_index=~/cluewebindex --new_index=new_indices/all_doc_index_post --indri_path=work_indri"



out = run_bash_command(summarization_command)
print(out,flush=True)
run_bash_command("rm -r summary_vectors/")
out = run_bash_command(vectors_command)
print(out,flush=True)
out = run_bash_command(updata_text_command)
print(out,flush=True)
run_bash_command("rm -r merged_indices/merged_index_post")
run_bash_command("rm -r new_indices/all_doc_index_post")
out = run_bash_command(rerank_command)
print(out,flush=True)
