from summarization.seo_experiment.utils import create_index,merge_indices,run_model,create_features_file
import sys,os,logging
from optparse import OptionParser

def merge(trectext_file,new_index,base_index,merged_index,home_path,indri_path):
    logger.info("creating index from trectext file")
    create_index(trectext_file,os.path.dirname(new_index),os.path.basename(new_index),home_path,indri_path)
    logger.info("index creation completed")
    logger.info("merging indices")
    merge_indices(merged_index,new_index,base_index,home_path,indri_path)
    logger.info("merging indices completed")

def rank(options):
    logger.info("creating features")
    features_file = create_features_file(options.features_dir, options.merged_index, options.queries_file,
                                         options.new_features_file, options.workingset_file, options.scripts_path)
    logger.info("features creation completed")
    logger.info("running ranking model on features file")
    run_model(features_file, options.home_path, options.java_path, options.jar_path, options.score_file,
              options.model_file)
    logger.info("ranking completed")

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode")
    parser.add_option("-k", "--trectext_file", dest="trectext_file")
    parser.add_option("-d", "--base_index", dest="base_index")
    parser.add_option("-t", "--new_index", dest="new_index")
    parser.add_option("-j", "--merged_index", dest="merged_index")
    parser.add_option("-p", "--home_path", dest="home_path")
    parser.add_option("-t", "--indri_path", dest="indri_path")
    parser.add_option("-a", "--script_path", dest="script_path")
    parser.add_option("-i", "-model_file", dest="model_file")
    parser.add_option("-a", "--workingset_file", dest="workingset_file")
    parser.add_option("-f", "--features_dir", dest="features_dir")
    parser.add_option("-w", "--new_features_file", dest="new_features_file")
    parser.add_option("-q", "--queries_file", dest="queries_file")
    parser.add_option("-e", "--scripts_path", dest="scripts_path")
    parser.add_option("-z", "--java_path", dest="java_path")
    parser.add_option("-c", "--jar_path", dest="jar_path")
    parser.add_option("-s", "--score_file", dest="score_file")

    (options, args) = parser.parse_args()
    mode = options.mode
    if mode=="merge":
        merge(options.trectext_file,options.new_index,options.base_index,options.merged_index,options.home_path,options.indri_path)
    elif mode=="rank":
        rank(options)
    elif mode=="all":
        merge(options.trectext_file, options.new_index, options.base_index, options.merged_index, options.home_path,
              options.indri_path)
        rank(options)
    else:
        logger.error("mode selection error!")
        sys.exit(1)

