from gen_utils import run_bash_command

for ref_index in [1,2,3,4]:
    number_of_top_docs = min(3,ref_index)
    prep_vectors_command = "python create_reference_docs_sentence_vectors.py --trec_file=trecs/trec_file_original_sorted.txt --trectext_file=data/documents.trectext --ref_index="+str(ref_index)+" --index=merged_indices/merged_index --sentences_out_file=data/ref_sentences_"+str(ref_index)+".txt --vectors_output_dir=sentence_ref_vectors_"+str(ref_index)+"/"

    summarization_command = "python summarization_process.py --mode=all --summary_input_file=input_data/input_paragraphs_transformer_"+str(ref_index)+".txt --sum_model=transformer --trectext_file=data/documents.trectext --trec_file=trecs/trec_file_original_sorted.txt --ref_index="+str(ref_index)+" --candidate_dir=~/textGen/summarization/summarization_pool/ --queries_file=data/queries_seo_exp.xml --model_file=/lv_local/home/sgregory/textGen/summarization/seo_experiment/bot_exp_utils/word2vec_model --summary_script_file=~/OpenNMT-py/translate.py --summary_output_file=summaries/output --number_of_top_docs="+str(number_of_top_docs)+" --sentences_vectors_dir=sentence_ref_vectors_"+str(ref_index)+"/ --documents_vectors_dir=asr_tfidf_vectors/ --suffix="+str(ref_index)+" --paragraph_vectors_dir=paragraph_vectors/"

    vectors_command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/lv_local/home/sgregory/indri-5.6/swig/obj/java/ -cp seo_summarization.jar PrepareTFIDFVectorsSummaryText merged_indices/merged_index/ input_data/all_data_transformer_"+str(ref_index)+".txt summaries/output_transformer_"+str(ref_index)+".txt summary_vectors_"+str(ref_index)+"/"

    updata_text_command = "python choose_summaries_for_replacement.py --doc_tfidf_dir=asr_tfidf_vectors/ --summaries_tfidf_dir=summary_vectors_"+str(ref_index)+"/ --queries_file=input_data/queries_transformer_"+str(ref_index)+".txt --summaries_file=summaries/output_transformer_"+str(ref_index)+".txt --input_data_file=input_data/all_data_transformer_"+str(ref_index)+".txt --trec_file=trecs/trec_file_original_sorted.txt --number_of_top_docs=3 --trectext_file=data/documents.trectext --new_trectext_file=data/updated_documents_"+str(ref_index)+".trectext --new_ws_file=data/updated_workingset --model_file=/lv_local/home/sgregory/textGen/summarization/wiki.en.bin"

    # rerank_command = "python ranking_process.py --mode=all --features_dir=Features_post_"+str(ref_index)+" --merged_index=merged_indices/merged_index_post --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_post_"+str(ref_index)+" --workingset_file=data/workingset_original --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_post_"+str(ref_index)+".txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_post_"+str(ref_index)+" --trectext_file=data/updated_documents_"+str(ref_index)+".trectext --home_path=~/ --base_index=~/cluewebindex --new_index=new_indices/all_doc_index_post_"+str(ref_index)+" --indri_path=work_indri"
    rerank_command = "python reranking_process.py --mode=all --features_dir=Features_post_"+str(ref_index)+"/ --merged_index=merged_indices/merged_index --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_post_"+str(ref_index)+" --workingset_file=data/workingset_original --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_post_"+str(ref_index)+".txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_post_"+str(ref_index)+" --trectext_file=data/updated_documents_"+str(ref_index)+".trectext --home_path=~/ --base_index=~/cluewebindex --new_index=new_indices/all_doc_index_post_"+str(ref_index)+" --indri_path=work_indri"

    # out = run_bash_command(prep_vectors_command)
    # print(out, flush=True)
    # out = run_bash_command(summarization_command)
    # print(out,flush=True)
    # # run_bash_command("rm -r summary_vectors/")
    # out = run_bash_command(vectors_command)
    # print(out,flush=True)
    # out = run_bash_command(updata_text_command)
    # print(out,flush=True)
    # # run_bash_command("rm -r merged_indices/merged_index_post")
    out = run_bash_command(rerank_command)
    print(out,flush=True)
