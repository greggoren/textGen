from summarization.seo_experiment.utils import read_trec_file,load_file,run_bash_command
import sys
def get_ref_docs(ranked_list,index):
    ref_docs={}
    for epoch in ranked_list:
        for query in ranked_list[epoch]:
            qid = str(int(query))+epoch
            ref_doc = ranked_list[query][epoch][index]
            ref_docs[qid]=ref_doc
    return ref_docs

def gather_docs_for_working_set(texts,starting_epoch,last_epoch,ref_docs):
    workingset_docs={}
    for doc in texts:
        epoch=doc.split("-")[1]
        if int(epoch)<starting_epoch:
            continue
        query=doc.split("-")[2]
        qid=str(int(query))+epoch
        if qid not in workingset_docs:
            workingset_docs[qid]=[]
        if doc==ref_docs[qid]:
            if int(epoch)==last_epoch:
                continue
            next_qid =str(int(query))+str(int(epoch)+1).zfill(2)
            if next_qid not in workingset_docs:
                workingset_docs[next_qid]=[]
            workingset_docs[next_qid].append(doc)
        else:
            workingset_docs[qid].append(doc)
    return workingset_docs




def create_working_set(ref_docs,texts,starting_epoch,last_epoch,workingset_fname):
    workingset_docs = gather_docs_for_working_set(texts,starting_epoch,last_epoch,ref_docs)
    with open(workingset_fname,"w") as out_working_set:
        for qid in workingset_docs:
            for i,doc in enumerate(workingset_docs[qid]):
                    out_working_set.write(qid+" Q0 "+doc+" "+str(i+1)+" "+str(-(i+1))+" dynamic_experiment\n")



def run_reranking(working_set_fname,fname_addition,trectext_fname):
    rerank_command = "python reranking_process.py --mode=all --features_dir=Features_" + fname_addition + "_post_" + str(
        ref_index) + "/ --merged_index=merged_indices/merged_index/ --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_" + fname_addition + "_post_" + str(
        ref_index) + " --workingset_file=data/" + working_set_fname + " --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_" + fname_addition + "_post_" + str(
        ref_index) + ".txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_" + fname_addition + "_post_" + str(
        ref_index) + " --trectext_file="+trectext_fname+" --home_path=~/ --base_index=merged_indices/merged_index --new_index=new_indices/dynamic_experiment_"+fname_addition+"_" + str(
        ref_index) + " --indri_path=work_indri"
    out = run_bash_command(rerank_command)
    print(out,flush=True)




if __name__=="__main__":
    for ref_index in ["1","2","3","4"]:
        trectext_file_prefix = sys.argv[1]
        trec_file = sys.argv[2]
        fname_addition = sys.argv[3]
        trectext_fname=trectext_file_prefix+"_"+ref_index+".trectext"
        texts = load_file(trectext_fname)
        ranked_lists = read_trec_file(trec_file)
        ref_docs = get_ref_docs(ranked_lists,int(ref_index))
        workingset_fname = "dynamic_experiment_workingset_"+ref_index+".txt"
        create_working_set(ref_docs,texts,7,8,workingset_fname)
        run_reranking(workingset_fname,fname_addition,trectext_fname)

