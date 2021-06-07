import argparse
import os
from os import listdir, walk
from os.path import isfile, join
import gzip
import nltk
import util
from nltk.tokenize import sent_tokenize
import spacy
import time
from sqlitedict import SqliteDict
import sqlite3 as lite
import sys
nlp = spacy.load("en_core_web_lg", disable=["ner"])
from itertools import islice, chain
import pickle as pkl
'''piped files'''
import util
## TODO: CTRL-F 'return -1' 'break'
## TODO: Line 93, remove debugging to include all files

'''usage: (goes in shell script create_amalgam_and_partitions.sh)
python3 -u AWS_code/create_amalgams.py \
-outDir 'AWS_code/bigjob/' \
-dataDir '/nlp/data/corpora/LDC/LDC2011T07/data/' \
-NUM_JOBS 3 \
> AWS_code/bigjob/combine_amalgams_create_sqlitedicts.stdout 2>&1
'''
## global argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help='example: out/')
parser.add_argument('-dataDir', required=True, help="example: 'data/'' or /'nlp/data/corpora/LDC/LDC2011T07/data/'")
parser.add_argument('-NUM_JOBS', type=int, required=True, help='example: 960')
parser.add_argument('-clip_len', type=int, default=225, required=False, help='example: 225')
args = parser.parse_args()

## global variables
sentences, trace, spacy_toks, spacy_pos, spacy_deps = [], [], [], [], []
sentences_dict, trace_dict = SqliteDict(args.outDir+'sentences.db'), SqliteDict(args.outDir+'trace.db')
spacy_toks_dict, spacy_pos_dict, spacy_deps_dict = SqliteDict(args.outDir+'spacy_toks.db'), SqliteDict(args.outDir+'spacy_pos.db'), SqliteDict(args.outDir+'spacy_deps.db')
job_partitions = []

# This function wraps util.pickle_load to lazily load the pickle files.
def lazy_pickle_load(fname):
    yield util.pickle_load(fname)

## Reads in a sample gigaword file (non-annotated)
def add_file(cur_sent_job, cur_trace_job, cur_spacy_toks_job, cur_spacy_pos_job, cur_spacy_deps_job):
    '''Reads in a sample gigaword file (non-annotated, 1 month of NYT)
    Args:
    Returns:
    Note:
    '''
    global args, sentences, trace, spacy_toks, spacy_pos, spacy_deps

    start = time.time()

    ##load data
    cur_sentences = lazy_pickle_load(cur_sent_job)
    cur_trace = lazy_pickle_load(cur_trace_job)
    cur_toks = lazy_pickle_load(cur_spacy_toks_job)
    cur_pos = lazy_pickle_load(cur_spacy_pos_job)
    cur_deps = lazy_pickle_load(cur_spacy_deps_job)

    ## extend global variables
    sentences.append(cur_sentences)
    trace.append(cur_trace)
    spacy_toks.append(cur_toks)
    spacy_pos.append(cur_pos)
    spacy_deps.append(cur_deps)

    extend_time = time.time() - start

    print('month added.. {}'.format(cur_sent_job[-18:-4]))
    # print('len(sentences): {} len(trace): {}'.format(len(cur_sentences), len(cur_trace)))
    # print('len(spacy_toks): {} len(spacy_pos): {} len(spacy_deps): {}'.format(len(cur_toks), len(cur_pos), len(cur_deps)))

    print('Time elapsed adding month to amalgamation: {}'.format(time.strftime("%H:%M:%S", time.gmtime(extend_time))))


def read_files():

    global args
    ## /nlp/data/corpora/LDC/LDC2011T07/data/
    ## sorted([join(rootdir+'/',fname) for (rootdir,subdir,fnames) in list(walk(dataDir))[1:] for fname in fnames])
    sent_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('sentences_job')])
    trace_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('trace_job')])
    spacy_toks_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('spacy_toks_job')])
    spacy_pos_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('spacy_pos_job')])
    spacy_deps_fnames = sorted([args.outDir+fname for fname in os.listdir(args.outDir) if fname.startswith('spacy_deps_job')])
    fnames_len = len(sent_fnames)
    print('\nlengths of filenames should match: ')
    print('\nsent_fnames {}, \ntrace_fnames {}, \nspacy_toks_fnames {}, \nspacy_pos_fnames {}, \nspacy_deps_fnames{}'.format(\
        fnames_len, \
        len(trace_fnames), \
        len(spacy_toks_fnames), \
        len(spacy_pos_fnames), \
        len(spacy_deps_fnames)))

    print('\n')
    for idx, fname in enumerate(sent_fnames):
        print('\nround {} - adding file {}... percentage processed {}'.format(idx, fname[-18:-4], (idx/fnames_len)*100))
        print('Current set of files being processed: \n{}, \n{}, \n{}, \n{} , \n{}'.format(sent_fnames[idx], trace_fnames[idx], spacy_toks_fnames[idx], spacy_pos_fnames[idx], spacy_deps_fnames[idx]))
        add_file(sent_fnames[idx], trace_fnames[idx], spacy_toks_fnames[idx], spacy_pos_fnames[idx], spacy_deps_fnames[idx])

    # print('\n--totals check in read_files()--')
    # print('len(sentences): {} len(trace): {}'.format(len(sentences), len(trace)))
    # print('len(spacy_toks): {}, len(spacy_pos): {}, len(spacy_deps): {}'.format(len(spacy_toks), len(spacy_pos), len(spacy_deps)))


def progress_in_batches(name, iterator, increment):
    count = 0
    while True:
        next_batch = list(islice(iterator, increment))
        if len(next_batch) > 0:
            count += len(next_batch)
            print('\nCreating {} SQLiteDict... number processed {}'.format(name, count))
            yield next_batch
        else:
            break

def create_sqlite_dicts():

    print('\ncreating sqlite dicts..')
    start = time.time()
    total_sentences = 0
    global sentences_dict, trace_dict, spacy_toks_dict, spacy_pos_dict, spacy_deps_dict, sentences, trace, spacy_toks, spacy_pos, spacy_deps

    ## build dicts
    flattened_sentences = enumerate(chain.from_iterable(chain.from_iterable(sentences)))
    for batch in progress_in_batches('sentences', flattened_sentences, 10000):
        update_value = [(idx,sent) for idx,sent in batch]
        total_sentences += len(update_value)
        sentences_dict.update(update_value)
    flattened_trace = enumerate(chain.from_iterable(chain.from_iterable(trace)))
    for batch in progress_in_batches('trace', flattened_trace, 10000):
        trace_dict.update([(idx,cur_trace) for idx,cur_trace in batch])
    flattened_spacy_toks = enumerate(chain.from_iterable(chain.from_iterable(spacy_toks)))
    for batch in progress_in_batches('spacy_toks', flattened_spacy_toks, 10000):
        spacy_toks_dict.update([(idx,cur_toks) for idx,cur_toks in batch])
    flattened_spacy_pos = enumerate(chain.from_iterable(chain.from_iterable(spacy_pos)))
    for batch in progress_in_batches('spacy_pos', flattened_spacy_pos, 10000):
        spacy_pos_dict.update([(idx,cur_pos) for idx,cur_pos in batch])
    flattened_spacy_deps = enumerate(chain.from_iterable(chain.from_iterable(spacy_deps)))
    for batch in progress_in_batches('spacy_deps', flattened_spacy_deps, 10000):
        spacy_deps_dict.update([(idx,cur_deps) for idx,cur_deps in batch])

    creating_time = time.time() - start
    print('Time elapsed creating sqlite dicts: {}'.format(time.strftime("%H:%M:%S", time.gmtime(creating_time))))

    # start = time.time()
    # print('\n---dict_stats---')
    # print('len(sentences_dict): {} len(reverse_sentences_dict): {} len(trace_dict): {}'.format(len(sentences_dict), len(reverse_sentences_dict), len(trace_dict)))
    # print('len(spacy_toks_dict): {} len(spacy_pos_dict): {} len(spacy_deps_dict): {}'.format(len(spacy_toks_dict), len(spacy_pos_dict), len(spacy_deps_dict)))
    # printing_time = time.time() - start
    # print('Time elapsed printing sqlite dicts: {}'.format(time.strftime("%H:%M:%S", time.gmtime(printing_time))))

    # print('printing the dictionary keys for one of the sqlitedicts: ')
    # print(sentences_dict.keys())

    ## commit dicts
    sentences_dict.commit()
    trace_dict.commit()
    spacy_toks_dict.commit()
    spacy_pos_dict.commit()
    spacy_deps_dict.commit()


    ## close dicts
    sentences_dict.close()
    trace_dict.close()
    spacy_toks_dict.close()
    spacy_pos_dict.close()
    spacy_deps_dict.close()

    return total_sentences



#####################
## Helper functions
#####################

def get_partition_slice(it, size, section):
    '''
    return in the indices for this slice (for the section requested)
    I could change this to only return the start and end index of each subarray instead of all the indices for that partition
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))[section]

def get_slice(it, size):
    '''
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))

if __name__ == '__main__':

    print('---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    start = time.time()

    ## reads in files created from the job array and combines them into an amalgamated index of each: 
    ## sentences_amalgam, trace_amalgam, and spacy_toks_amalgam, spacy_pos_amalgam, spacy_deps_amalgam
    read_files()

    ## create sqllite dicts, saves output files (including spacy annotations)
    total_sentences = create_sqlite_dicts()

    ## get slices for embed_and_filter
    npartitions, remainder = divmod(total_sentences, args.NUM_JOBS)
    print('\nnpartitions {}, remainder {}'.format(npartitions, remainder))
    partitions = get_slice(list(range(total_sentences)), npartitions+remainder)
    print('len(paritions) calculated for {} NUM_JOBS: {} '.format(args.NUM_JOBS, len(partitions)))

    with open(args.outDir+"job_slices.pkl", "wb") as fout:
        for part in partitions:
            job_partitions.append((part[0], part[-1]))
        pkl.dump(job_partitions,fout)
    fout.close()

    elapsed_time = time.time() - start

    print('Time elapsed in main: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    print('\nfinished processing files...')


