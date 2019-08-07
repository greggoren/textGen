import sys, getopt,os,logging
from optparse import OptionParser

import subprocess

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')

def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out


def run_decode_script(decode_script):
    command = "./"+decode_script
    logger.info("Running script "+command)
    out = run_bash_command(command)
    logger.info(out)




def create_checkpoint_file(train_dir,wanted_checkpoint):
    backup_file = train_dir+"checkpoint_bkup"
    working_file = train_dir+"checkpoint"
    if os.path.isfile(working_file):
        run_bash_command("mv "+working_file+" "+backup_file)
    run_bash_command("touch "+train_dir+"checkpoint")
    run_bash_command('echo "model_checkpoint_path: "model.ckpt-'+wanted_checkpoint+'"" >> '+working_file)
    run_bash_command('tail -n2 '+backup_file+ ' >> '+working_file)

def retrieve_all_checkpoints(train_dir):
    checkpoints=[]
    with open(train_dir+"checkpoint") as file:
        first = True
        for line in file:
            if first:
                first=False
                continue
            checkpoint = line.split("-")[1].replace('"','').rstrip()
            checkpoints.append(checkpoint)
        return checkpoints


def filter_checkpoints(checkpoints,lower,upper):
    reduced = []
    for c in checkpoints:
        if int(c)>=int(lower) and int(c)<=int(upper):
            reduced.append(c)
    return reduced



if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # opts, args = getopt.getopt(sys.argv, "mode:train_dir:decode_script:")
    mode = ''
    train_dir=''
    decode_script=''
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode",
                      help="set running mode")
    parser.add_option("-td", "--train_dir",dest="train_dir")
    parser.add_option("-ds", "--decode_script",dest = "decode_script")
    (options, args) = parser.parse_args()
    mode = options.mode
    train_dir=options.train_dir
    decode_script = options.decode_script
    if mode=="ALL" or mode=="all":
        checkpoints = retrieve_all_checkpoints(train_dir)
        for checkpoint in checkpoints:
            create_checkpoint_file(train_dir,checkpoint)
            run_decode_script(decode_script)
    if "range" in mode.lower():
        lower_checkpoint = mode.split("_")[1]
        upper_checkpoint = mode.split("_")[2]
        checkpoints = retrieve_all_checkpoints(train_dir)
        checkpoints=filter_checkpoints(checkpoints,lower_checkpoint,upper_checkpoint)
        for checkpoint in checkpoints:
            create_checkpoint_file(train_dir, checkpoint)
            run_decode_script(decode_script)
    if "specific" in mode.lower():
        checkpoint = mode.split("_")[1]
        create_checkpoint_file(train_dir, checkpoint)
        run_decode_script(decode_script)



