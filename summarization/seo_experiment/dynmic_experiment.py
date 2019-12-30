from summarization.seo_experiment.utils import read_trec_file,load_file,run_bash_command,order_trec_file
import sys,os


def get_ref_docs(ranked_list,index):
    ref_docs={}
    for epoch in ranked_list:
        for query in ranked_list[epoch]:
            qid = str(int(query))+epoch
            ref_doc = ranked_list[epoch][query][index]
            ref_docs[qid]=ref_doc
    return ref_docs

def former_doc(doc):
    return "-".join([doc.split("-")[0],str(int(doc.split("-")[1])-1).zfill(2),doc.split("-")[2],doc.split("-")[3]])

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
            if int(epoch)>=last_epoch:
                former_qid = str(int(query)) + str(int(epoch) - 1).zfill(2)
                if former_doc(doc) == ref_docs[former_qid]:
                    continue
                workingset_docs[qid].append(doc)
            else:
                next_qid =str(int(query))+str(int(epoch)+1).zfill(2)
                if next_qid not in workingset_docs:
                    workingset_docs[next_qid]=[]
                workingset_docs[next_qid].append(doc)
        else:
            if int(epoch)==starting_epoch:
                continue
            former_qid = str(int(query)) + str(int(epoch) - 1).zfill(2)
            if former_qid not in ref_docs:
                workingset_docs[qid].append(doc)
                continue
            if former_doc(doc) == ref_docs[former_qid]:
                continue
            workingset_docs[qid].append(doc)
    return workingset_docs




def create_working_set(ref_docs,texts,starting_epoch,last_epoch,workingset_fname):
    workingset_docs = gather_docs_for_working_set(texts,starting_epoch,last_epoch,ref_docs)
    with open(workingset_fname,"w") as out_working_set:
        for qid in workingset_docs:
            for i,doc in enumerate(workingset_docs[qid]):
                    out_working_set.write(qid+" Q0 "+doc+" "+str(i+1)+" "+str(-(i+1))+" dynamic_experiment\n")
    return workingset_docs



def run_reranking(working_set_fname, fname_addition, starting_epoch, trectext_fname):
    current_fname_addition= fname_addition + "_" + str(starting_epoch)

    rerank_command = "python reranking_process.py --mode=all --features_dir=Features_" + current_fname_addition + "_post_" + str(
        ref_index) + "/ --merged_index=merged_indices/merged_index --queries_file=data/queries_seo_exp.xml --new_features_file=final_features_dir/features_" + current_fname_addition + "_post_" + str(
        ref_index) + " --workingset_file=" + working_set_fname + " --scripts_path=scripts/ --java_path=jdk1.8.0_181 --jar_path=scripts/RankLib.jar --score_file=scores/scores_" + current_fname_addition + "_post_" + str(
        ref_index) + ".txt --model_file=rank_models/model_bot --trec_file=trecs/trec_file_" + current_fname_addition + "_post_" + str(
        ref_index) + " --trectext_file=" + trectext_fname +" --home_path=~/ --base_index=~/cluewebindex --new_index=new_indices/dynamic_experiment_" + current_fname_addition + "_" + str(
        ref_index) + " --indri_path=work_indri"
    out = run_bash_command(rerank_command)

    print(out,flush=True)
    return "trecs/trec_file_" + current_fname_addition + "_post_" + str(ref_index)


def fix_xml_file(fname):
    with open(fname,encoding="utf-8",errors="ignore") as f:
        lines = f.readlines()
        lines.insert(0,"<DATA>\n")
        lines.append("</DATA>\n")
    fixed_fname = "data/fixed_trectext_for_run.trectext"
    with open(fixed_fname,"w",encoding="utf-8",errors="ignore") as out:
        out.writelines(lines)
    return fixed_fname



def create_trectext_dynamic(texts,original_texts,workingset_docs,trec_fname):
    with open(trec_fname,"w") as f:
        for query in workingset_docs:
           query_epoch = query[-1].zfill(2)
           for doc in workingset_docs[query]:
               doc_epoch = doc.split("-")[1]
               if doc_epoch!=query_epoch:
                   text = texts[doc]
               else:
                   text = original_texts[doc]
               f.write("<DOC>\n")
               f.write("<DOCNO>"+doc+"</DOCNO>\n")
               f.write("<TEXT>\n")
               f.write(bytes(str(text), 'cp1252', "ignore").decode('utf-8', 'ignore').rstrip()+"\n")
               f.write("</TEXT>\n")
               f.write("</DOC>\n")


def append_to_file(source,target):
    with open(source) as input:
        lines = input.readlines()
        with open(target,"a") as output:
            output.writelines(lines)


if __name__=="__main__":
    trectext_file_prefix = sys.argv[1]
    trec_file = sys.argv[2]
    fname_addition = sys.argv[3]
    starting_epoch = int(sys.argv[4])
    for ref_index in ["1","2","3","4"]:
        final_trec_name = "trecs/trec_file_" + fname_addition + "_post_" + str(ref_index)
        if os.path.exists(final_trec_name):
            os.remove(final_trec_name)
        for r in range(starting_epoch,8):
            trectext_fname=trectext_file_prefix+"_"+ref_index+".trectext"
            trectext_fname_new=trectext_file_prefix+"_"+ref_index+"_"+str(r)+"_new.trectext"
            trectext_file_for_read = fix_xml_file(trectext_fname)
            texts = load_file(trectext_file_for_read)
            original_texts = load_file("data/documents.trectext")
            ranked_lists = read_trec_file(trec_file)
            ref_docs = get_ref_docs(ranked_lists,int(ref_index))
            workingset_fname = "data/dynamic_experiment_workingset_"+ref_index+"_"+str(r)+".txt"
            workingset_docs = create_working_set(ref_docs,texts,r,r+1,workingset_fname)
            create_trectext_dynamic(texts,original_texts,workingset_docs,trectext_fname_new)
            tmp_trec_file = run_reranking(workingset_fname,fname_addition,r,trectext_fname_new)
            append_to_file(tmp_trec_file,final_trec_name)
            os.remove(tmp_trec_file)
        order_trec_file(final_trec_name)
        os.remove(final_trec_name)

