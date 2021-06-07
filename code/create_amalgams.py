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
## TODO: CTRL-F 'return -1' 'break'
## TODO: Line 111, remove debugging include all files

'''usage: (goes in shell script create_amalgam_and_partitions.sh)
python3 -u code/create_amalgams.py \
-outDir 'AWS_code/1out/' \
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

## Reads in a sample gigaword file (non-annotated)
def add_file(gigaword_fname):
    '''Reads in a sample gigaword file (non-annotated, 1 month of NYT)
    Args:
    Returns:
    Note:
    '''
    global args, sentences, trace, spacy_toks, spacy_pos, spacy_deps
    counter = 0
    cur_sentences, cur_trace = [], []
    cur_ndocs, cur_nsents = 0, 0
    cur_text, doc_id, begin_tok = '', '', ''

    start = time.time()

    with gzip.open(gigaword_fname, 'rt') as f: 
        for i, line in enumerate(f):
            ## Saves document id
            if line.startswith('<DOC id='):
                doc_id = line.strip()[9:30]
                cur_ndocs += 1
            elif line.startswith('<P>'):
                begin_tok = line.strip()
            elif line.startswith('</P>'):
                continue
            elif line.startswith('</TEXT>'):
                if len(cur_text) != 0:
                    tok_sents = sent_tokenize(cur_text)
                    cur_nsents += (len(tok_sents))
                    cur_sentences.extend([sent.strip() for sent in tok_sents])
                    cur_trace.extend([doc_id for sent_id in range(len(tok_sents))])                 
                    cur_text, begin_tok = '', ''
                    # test for add 10 sents per file for debugging
                    # if cur_nsents > 10:
                    #     break
                    # test for specific num of files
                    # if counter >2:
                    #     break
                    counter += 1                                     
            else:
                if begin_tok == '<P>':
                    cur_text += " " + line.strip()

    read_time = time.time() - start

    start = time.time()
    ## get cur spacy annotations
    spacy_docs = list(nlp.pipe(cur_sentences)) 
    cur_toks = [[tok.text for tok in doc][:args.clip_len] for doc in spacy_docs]
    cur_pos = [[tok.pos_ for tok in doc][:args.clip_len] for doc in spacy_docs]
    cur_deps = [[tok.dep_ for tok in doc][:args.clip_len] for doc in spacy_docs]
    spacy_time = time.time() - start

    ## extend global variables
    sentences.extend(cur_sentences)
    trace.extend(cur_trace)
    spacy_toks.extend(cur_toks)
    spacy_pos.extend(cur_pos)
    spacy_deps.extend(cur_deps)

    print('cur_ndocs: {}  cur_nsents: {}'.format(cur_ndocs, cur_nsents))
    print('len(sentences): {} \nlen(trace): {} \nlen(spacy_toks): {} \nlen(spacy_pos): {} \nlen(spacy_deps): {}'.format(len(cur_sentences), len(cur_trace), len(cur_toks), len(cur_pos), len(cur_deps)))
    print('Time elapsed reading file for {}:\t{}'.format(gigaword_fname[-17:], time.strftime("%H:%M:%S", time.gmtime(read_time))))
    print('Time elapsed generating spacy annotations for {}:\t{}'.format(gigaword_fname[-17:], time.strftime("%H:%M:%S", time.gmtime(spacy_time))))

def read_files():

    global args
    ## /nlp/data/corpora/LDC/LDC2011T07/data/
    ## sorted([join(rootdir+'/',fname) for (rootdir,subdir,fnames) in list(walk(dataDir))[1:] for fname in fnames])
    fnames = sorted([join(args.dataDir,subdir,fname) for subdir in listdir(args.dataDir) for fname in listdir(args.dataDir+subdir)]) 
    fnames_len = len(fnames)
    print('\nnfiles: ', fnames_len)
    # print('\nfiles: ', fnames)

    print('\n')
    for idx, fname in enumerate(fnames):
        print('\nround {} processing {}... percentage processed {}'.format(idx, fname[-17:], (idx/fnames_len)*100))
        add_file(fname)

    # print('\n--totals check in read_files()--')
    # print('len(sentences): {} len(trace): {}'.format(len(sentences), len(trace)))
    # print('len(spacy_toks): {}, len(spacy_pos): {}, len(spacy_deps): {}'.format(len(spacy_toks), len(spacy_pos), len(spacy_deps)))


def progress_in_batches(name, iterator, total_length, increment):
    count = 0
    while True:
        next_batch = list(islice(iterator, increment))
        if len(next_batch) > 0:
            count += len(next_batch)
            print('\nCreating SQLiteDict {}... percentage processed {}'.format(name, (count/total_length)*100))
            yield next_batch
        else:
            break

def create_sqlite_dicts():

    print('\ncreating sqlite dicts..')
    start = time.time()
    global sentences_dict, trace_dict, spacy_toks_dict, spacy_pos_dict, spacy_deps_dict, sentences, trace, spacy_toks, spacy_pos, spacy_deps

    ## build dicts
    sqlite_sent_start = time.time()
    for batch in progress_in_batches('sentences', enumerate(sentences), len(sentences), 10000):
        sqlite_start = time.time()
        sentences_dict.update([(idx,sent) for idx,sent in batch]) ##I'm worried about this not having unique keys
        sqlite_time = time.time() - sqlite_start
        print('Time elapsed creating sqlite dictionary for a single batch process: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_time))))
    sqlite_sent_time = time.time() - sqlite_sent_start
    print('Time elapsed creating sentences sqlite dictionaries: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_sent_time))))
    sqlite_trace_start = time.time()
    for batch in progress_in_batches('trace', enumerate(trace), len(trace), 10000):
        trace_dict.update([(idx,cur_trace) for idx,cur_trace in batch])
    sqlite_trace_time = time.time() - sqlite_trace_start
    print('Time elapsed creating trace sqlite dictionaries: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_trace_time))))
    sqlite_spacy_tok_start = time.time()
    for batch in progress_in_batches('spacy_toks', enumerate(spacy_toks), len(spacy_toks), 10000):
        spacy_toks_dict.update([(idx,cur_toks) for idx,cur_toks in batch])
    sqlite_spacy_tok_time - time.time() - sqlite_spacy_tok_start
    print('Time elapsed creating spacy tok sqlite dictionaries: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_spacy_tok_time))))
    sqlite_spacy_pos_start = time.time()
    for batch in progress_in_batches('spacy_pos', enumerate(spacy_pos), len(spacy_pos), 10000):
        spacy_pos_dict.update([(idx,cur_pos) for idx,cur_pos in batch])
    sqlite_spacy_pos_time = time.time() - sqlite_spacy_pos_start
    print('Time elapsed creating spacy pos sqlite dictionaries: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_spacy_pos_time))))
    sqlite_spacy_deps_start = time.time()
    for batch in progress_in_batches('spacy_deps', enumerate(spacy_deps), len(spacy_deps), 10000):
        spacy_deps_dict.update([(idx,cur_deps) for idx,cur_deps in batch])
    sqlite_spacy_deps_time = time.time() - sqlite_spacy_deps_start
    print('Time elapsed creating spacy deps sqlite dictionaries: {}'.format(time.strftime("%H:%M:%S", time.gmtime(sqlite_spacy_deps_time))))

    creating_time = time.time() - start
    print('Time elapsed creating all sqlite dicts: {}'.format(time.strftime("%H:%M:%S", time.gmtime(creating_time))))

    # start = time.time()
    # print('\n---dict_stats---')
    # print('len(sentences_dict): {} len(reverse_sentences_dict): {} len(trace_dict): {}'.format(len(sentences_dict), len(reverse_sentences_dict), len(trace_dict)))
    # print('len(spacy_toks_dict): {} len(spacy_pos_dict): {} len(spacy_deps_dict): {}'.format(len(spacy_toks_dict), len(spacy_pos_dict), len(spacy_deps_dict)))
    # printing_time = time.time() - start
    # print('Time elapsed printing sqlite dicts: {}'.format(time.strftime("%H:%M:%S", time.gmtime(printing_time))))

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

    ## reads in files, creates amalgamated index
    read_files()

    ## create sqllite dicts, saves output files (including spacy annotations)
    create_sqlite_dicts()

    ## get slices for embed_and_filter
    npartitions, remainder = divmod(len(sentences), args.NUM_JOBS)
    print('\nnpartitions {}, remainder {}'.format(npartitions, remainder))
    partitions = get_slice(list(range(len(sentences))), npartitions+remainder)
    print('len(paritions) calculated for {} NUM_JOBS: {} '.format(args.NUM_JOBS, len(partitions)))

    with open(args.outDir+"job_slices.pkl", "wb") as fout:
        for part in partitions:
            job_partitions.append((part[0], part[-1]))
        pkl.dump(job_partitions,fout)
    fout.close()

    elapsed_time = time.time() - start

    print('Time elapsed in main: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    print('\nfinished processing files...')



