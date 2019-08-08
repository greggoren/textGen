import os
from t2t.utils import run_bash_command
import time
from optparse import OptionParser
from multiprocessing import Pool
from functools import partial

def run_bleu(reference,script,translation):
    out = run_bash_command(script+" --translation="+translation+" --reference="+reference)

    for line in str(out).split("\n"):
        if "BLEU_uncased" in line:
            score = float(line.split()[2].rstrip())
            print(score)
            return score,translation




def calculate_bleu_multiprocess(translations_dir,reference_file,bleu_script_bin,write_flag,processes):
    files = [file for file in os.listdir(translations_dir)]
    if write_flag:
        ts = time.time()
        bleu_results_file = open("BLEU_RESULTS_"+str(ts),
                                 'w')
    func = partial(run_bleu,reference_file,bleu_script_bin)
    with Pool(processes=processes) as pool:
        results = pool.map(func,files)
        if write_flag:
            for result in results:
                bleu_results_file.write(result[1]+"\t"+result[0]+"\n")
            bleu_results_file.close()
        return results

def calculate_bleu(translations_dir,reference_file,bleu_script_bin,write_flag):
    files = [file for file in os.listdir(translations_dir)]
    files = sorted(files)
    bleus = []
    if write_flag:
        ts = time.time()
        bleu_results_file = open("BLEU_RESULTS_"+str(ts),
                                 'w')
    for file in files:
        translation_file = translations_dir+file
        score,_ = run_bleu(reference_file,bleu_script_bin,translation_file)
        bleus.append(score)
        if write_flag:
            bleu_results_file.write(file+"\t"+str(score)+"\n")
    if write_flag:
        bleu_results_file.close()

    return bleus



if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-m", "--metric", dest="metric",
                      help="set running mode")
    parser.add_option("-r", "--reference", dest="reference")
    parser.add_option("-t", "--translations_dir", dest="translations_dir")
    parser.add_option("-s", "--bleu_script", dest="bleu_script")
    parser.add_option("-w", "--write", dest="write",action="store_true")
    parser.add_option("-n", "--no_write", dest="write",action="store_false")
    parser.add_option("-k", "--multiproccess", dest="multi",default='0')
    (options, args) = parser.parse_args()
    metric = options.metric
    translation_dir = options.translations_dir
    bleu_script_path = options.bleu_script
    reference_file = options.reference
    write_flag = options.write
    if metric.lower()=="bleu":
        if options.multi=='0':
            calculate_bleu(translation_dir,reference_file,bleu_script_path,write_flag)
        else:
            calculate_bleu_multiprocess(translation_dir,reference_file,bleu_script_path,write_flag,int(options.multi))
