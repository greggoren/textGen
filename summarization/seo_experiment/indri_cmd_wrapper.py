import sys,os,logging
from optparse import OptionParser



if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode",
                      help="set running mode")
    parser.add_option("-k", "--train_dir", dest="train_dir")
    parser.add_option("-d", "--decode_script", dest="decode_script")
    parser.add_option("-t", "--translations_dir", dest="translations_dir")
    parser.add_option("-a", "--model", dest="model")
    parser.add_option("-i", "--data_dir", dest="data_dir")
    (options, args) = parser.parse_args()