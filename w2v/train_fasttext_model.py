#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.test.utils import get_tmpfile
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments

    if len(sys.argv) < 3:
        print
        globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    model = FastText(size=300, window=8, min_count=5)
    model.build_vocab(corpus_path=inp)
    model.train(sentences=LineSentence(inp),epochs=5,total_examples=model.corpus_count)
    # trim unneeded model memory = use (much) less RAM
    # model.init_sims(replace=True)
    fname = get_tmpfile(outp)
    model.save(fname)
