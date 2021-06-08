import gzip, json, sys
import numpy as np
import time
import argparse
import os
import util
from sqlitedict import SqliteDict

'''usage: (goes in shell script)
python3 -u code/create_word_index.py \
-outDir 'betatest/out/'
'''


'''global argparse'''
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help="Directory where all outfiles will be written to. Example: 'out/'")
args = parser.parse_args()

'''global variables'''
data = open(args.outDir+"shapes.txt","r")
shapes = data.read().splitlines()
shapes_dict = {}
for item in shapes: 
    fname,length = item.split(" ") 
    shapes_dict[fname]= int(length)
                      

if __name__ == '__main__':

    ## initialize the amalgamated words dict
    words = SqliteDict(args.outDir+"words.db")
    running_len = 0 

    print('---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    words_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('words_job')])
    print('\nnwords_fnames: ', len(words_fnames))
    for fname in words_fnames:
        print(fname)

    start = time.time()
    
    for words_fname in words_fnames:
        amalgamate_start = time.time()
        cur_len = shapes_dict[words_fname] 
        cur_words_dict = SqliteDict(words_fname)
        idx_iter = range(running_len, running_len+cur_len)
        values_iter = iter(cur_words_dict.values())
        words.update([(idx, word_tuple) for idx, word_tuple in zip(idx_iter, values_iter)])
        words.commit()
        cur_words_dict.close()
        running_len += cur_len
        cur_amalgamate_time = time.time() - amalgamate_start
        print('\nTime elapsed adding file %s to words_dict:\t%s' % (words_fname, time.strftime("%H:%M:%S", time.gmtime(cur_amalgamate_time))))

    print_start = time.time()
    print('\nlen(words): {}'.format(len(words)))
    print_time = time.time() - print_start

    words.close()
    elapsed_time = time.time() - start

    print('\n--stats--')
    print('\nTime elapsed creating amalgamated word index:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

