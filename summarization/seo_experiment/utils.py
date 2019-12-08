import os
import re
import javaobj

from gen_utils import run_bash_command,run_command
import xml.etree.ElementTree as ET
from  lxml import etree

def create_features_file_diif(features_dir, base_index_path,new_index_path,queries_file, new_features_file, working_set_file, scripts_path):
    """
    Creates  a feature file via a given index and a given working set file
    """
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(os.path.dirname(new_features_file)):
        os.makedirs(os.path.dirname(new_features_file))
    # command= scripts_path+"LTRFeatures "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+working_set_file + ' -workingSetFormat=trec'
    command="~/jdk1.8.0_181/bin/java -Djava.library.path=/lv_local/home/sgregory/indri-5.6/swig/obj/java/ -cp seo_summarization.jar LTRFeatures "+base_index_path+" "+new_index_path+" data/stopWordsList data/working_comp_queries.txt "+working_set_file+" "+features_dir
    print(command)
    out = run_bash_command(command)
    print(out)
    # run_bash_command("mv doc*_* "+features_dir)
    command = "perl " + scripts_path + "generate.pl " + features_dir + " " + working_set_file
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv featureID "+os.path.dirname(new_features_file))
    return new_features_file


def create_features_file(features_dir, index_path, queries_file, new_features_file, working_set_file, scripts_path):
    """
    Creates  a feature file via a given index and a given working set file
    """
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(os.path.dirname(new_features_file)):
        os.makedirs(os.path.dirname(new_features_file))
    command= scripts_path+"LTRFeatures "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+working_set_file + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl " + scripts_path + "generate.pl " + features_dir + " " + working_set_file
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv featureID "+os.path.dirname(new_features_file))
    return new_features_file

def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch = doc.split("-")[1]
            query = doc.split("-")[2]
            if epoch not in stats:
                stats[epoch]={}
            if query not in stats[epoch]:
                stats[epoch][query]=[]
            stats[epoch][query].append(doc)
    return stats

def read_raw_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            query = line.split()[0]
            if query not in stats:
                stats[query]=[]
            stats[query].append(doc)
    return stats


def create_trectext(document_text,trec_text_name,working_set_name):
    """
    creates trectext document from a given text file
    """
    f= open(trec_text_name,"w",encoding="utf-8")
    query_to_docs = {}
    for document in document_text:

        text = document_text[document]
        query = document.split("-")[2]
        if not query_to_docs.get(query,False):
            query_to_docs[query]=[]
        query_to_docs[query].append(document)

        f.write('<DOC>\n')
        f.write('<DOCNO>' + document + '</DOCNO>\n')
        f.write('<TEXT>\n')
        f.write(text.rstrip())
        f.write('\n</TEXT>\n')
        f.write('</DOC>\n')
    f.close()
    f = open(working_set_name, 'w')
    for query, docnos in query_to_docs.items():
        i = 1
        for docid in docnos:
            f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
            i += 1

    f.close()
    return trec_text_name



def create_index(trec_text_file,index_path,new_index_name,home_path = '/home/greg/',indri_path = "indri_test"):
    """
    Parse the trectext file given, and create an index.
    """
    indri_build_index = home_path+'/'+indri_path+'/bin/IndriBuildIndex'
    corpus_path = trec_text_file
    corpus_class = 'trectext'
    memory = '1G'
    index = index_path+"/"+new_index_name
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    stemmer =  'krovetz'
    if not  os.path.exists(home_path+"/"+index_path):
        os.makedirs(home_path+"/"+index_path)
    command = indri_build_index + ' -corpus.path=' + corpus_path + ' -corpus.class=' + corpus_class + ' -index=' + index + ' -memory=' + memory + ' -stemmer.name=' + stemmer
    print("##Running IndriBuildIndex command ="+command+"##",flush=True)
    out=run_bash_command(command)
    print("IndriBuildIndex output:"+str(out),flush=True)
    return index


def merge_indices(merged_index,new_index_name, base_index, home_path ='/home/greg/', indri_path ="indri_test"):
    """
    merges two different indri indices into one
    """
    # new_index_name = home_path +'/' + index_path +'/' + new_index_name
    if not os.path.exists(os.path.dirname(merged_index)):
        os.makedirs(os.path.dirname(merged_index))
    command = home_path+"/"+indri_path+'/bin/dumpindex '+merged_index +' merge ' + new_index_name + ' ' + base_index
    print("##merging command:",command+"##",flush=True)
    out=run_bash_command(command)
    print("merging command output:"+str(out),flush=True)
    return new_index_name


def create_trec_eval_file(results,trec_file):
    if not os.path.exists(os.path.dirname(trec_file)):
        os.makedirs(os.path.dirname(trec_file))
    trec_file_access = open(trec_file, 'w')
    for query in results:
        for doc in results[query]:
            trec_file_access.write(query+ " Q0 " + doc + " " + str(0) + " " + str(results[query][doc]) + " summarizarion_task\n")
    trec_file_access.close()
    return trec_file

# def create_trec_eval_file_rerank(results,trec_file):
#     if not os.path.exists(os.path.dirname(trec_file)):
#         os.makedirs(os.path.dirname(trec_file))
#     trec_file_access = open(trec_file, 'w')
#     for doc in results:
#         query = str(int(doc.split("-")[2]))+doc.split("-")[1]
#         trec_file_access.write(query+ " Q0 " + doc + " " + str(0) + " " + str(results[doc]) + " summarizarion_task\n")
#     trec_file_access.close()
#     return trec_file


def order_trec_file(trec_file):
    final = trec_file.replace(".txt", "")
    final+="_sorted.txt"
    command = "sort -k1,1n -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final

def retrieve_scores(test_indices, queries,score_file):
    results = {}
    with open(score_file) as scores:
        for i,score in enumerate(scores):
            query = queries[i]
            doc = test_indices[i]
            if query not in results:
                results[query]={}
            results[query][doc]=float(score.split()[2].rstrip())
        return results


def create_index_to_doc_name_dict(data_set_file):
    doc_name_index={}
    index = 0
    with open(data_set_file) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
        return doc_name_index

def create_index_to_query_dict(data_set_file):
    query_index={}
    index = 0
    with open(data_set_file) as ds:
        for line in ds:
            rec = line.split()
            query = rec[1].split(":")[1]
            query_index[index] = query
            index += 1
        return query_index

def run_model(test_file,home_path,java_path,jar_path,score_file,model_path):
    java_path = home_path+"/"+java_path+"/bin/java"
    if not os.path.exists(os.path.dirname(score_file)):
        os.makedirs(os.path.dirname(score_file))
    features = test_file
    run_bash_command('touch ' + score_file)
    command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
    out = run_bash_command(command)
    print(str(out))
    return score_file


def get_past_winners(ranked_lists,epoch,query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = str(iteration+1).zfill(2)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners

def reverese_query(qid):
    epoch = str(qid)[-2:]
    query = str(qid)[:-2].zfill(3)
    return epoch,query

def run_summarization_model(script_file,model_file,input_file,output_file,**kwargs):
    """
     cmd example:
     nohup python ~/OpenNMT-py/translate.py --replace_unk  -beam_size 10 --model ~/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt
      --src input_transformer.txt --output transformer_real_par2.txt
      --batch_size 1  -min_length 1  -gpu 0 &
    """
    command = "python "+script_file+" --replace_unk  -beam_size 10 --model "+model_file+" --src "+input_file+" --output "+output_file+" --batch_size 1 -gpu 0 "
    for key, value in kwargs.items():
        command+="--"+key+" "+value+" "
    print("##Running summarization command: "+command+"##",flush=True)
    out = run_bash_command(command)
    print("Summarization output= "+str(out),flush=True)

def load_file(filename):
    parser = etree.XMLParser(recover=True)
    tree = ET.parse(filename,parser=parser)
    root = tree.getroot()
    docs={}
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
            else:
                docs[name]=att.text
    return docs

def get_java_object(obj_file):
    with open(obj_file, 'rb') as fd:
        obj = javaobj.load(fd)
        return obj

def clean_texts(text):
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("$", "")
    text = text.replace("%", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace(";", " ")
    text = text.replace("`", "")
    text = text.replace("'", "")
    text = text.replace("@", " ")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = text.replace("/", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<t>","")
    text = text.replace("</t>","")
    return text.lower()
