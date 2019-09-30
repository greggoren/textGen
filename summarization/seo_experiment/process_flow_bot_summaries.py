from gen_utils import run_bash_command

for ref_index in [1,2,3,4]:
    number_of_top_docs = str(min(3,ref_index))
    raw_summary_command = "python create_raw_summary_ds.py "+str(ref_index)
    out = run_bash_command(raw_summary_command)
    print(out,flush=True)
    bot_summaries_command = "python bot_execution_summaries.py --mode=all --index_path=merged_indices/merged_index --raw_ds_out=data/raw_bot_summary_"+str(ref_index)+".txt --ref_index="+str(ref_index)+" --top_docs_index="+number_of_top_docs+" --doc_tfidf_dir=asr_tfidf_vectors/ --sentences_tfidf_dir=sentences_tfidf_vectors_"+str(ref_index)+"/ --summary_tfidf_dir=summary_vectors_"+str(ref_index)+"/ --queries_file=data/queries_seo_exp.xml --scores_dir=scores_summary_bot_"+str(ref_index)+"/ --trec_file=trecs/trec_file_original_sorted.txt --sentence_trec_file=trecs/bot_summary_trec_file_"+str(ref_index)+".txt --output_feature_files_dir=Features_bot_summary_"+str(ref_index)+"/ --output_final_feature_file_dir=features_bot_summary_"+str(ref_index)+"/ --trectext_file=data/documents.trectext --new_trectext_file=data/bot_summary_documents_"+str(ref_index)+".trectext --model_file=bot_exp_utils/word2vec_model --svm_model_file=bot_exp_utils/harmonic_competition_model --workingset_file=data/workingset_bot_summary"
    out = run_bash_command(bot_summaries_command)
    print(out,flush=True)
    rerank_command = "python reranking_process.py --mode=all --features_dir=Features_bot_summaries_post_"+str(ref_index)+"/ --merged_index=merged_indices/merged_index/ --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_bot_summaries_post_"+str(ref_index)+" --workingset_file=data/workingset_original --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_bot_summaries_post_"+str(ref_index)+".txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_bot__summary_post_"+str(ref_index)+" --trectext_file=data/updated_documents_"+str(ref_index)+".trectext --home_path=~/ --base_index=~/cluewebindex --new_index=new_indices/all_doc_index_post_bot_summary_"+str(ref_index)+" --indri_path=work_indri"
    out = run_bash_command(rerank_command)
    print(out,flush=True)
