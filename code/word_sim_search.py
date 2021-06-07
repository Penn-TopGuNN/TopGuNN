import sys, csv 
from tqdm import tqdm
import util
from collections import Counter
import time
from itertools import islice
import pdb
import os
import argparse
import nlp_analysis
import annoy
from annoy import AnnoyIndex
# import faiss
# from faiss import normalize_L2
import textwrap
from textwrap import indent
import time
import numpy as np
from sqlitedict import SqliteDict
## NOTE: device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ##.to(device)
## TODO: change name of stdout file, csv file, and annoy_index file to match the run_id (got to shell script)
## TODO: uncomment out annoy in imports when running faiss and vice versa when running annoy (see lines 11-14)

'''usage if running queries using ANNOY: (goes in shell script)
python3 -u AWS_code/word_sim_search.py \
-outDir 'AWS_code/8out_5mo/' \
-top_n 50 -num_trees 200 \
--ANNOY \
-sentence_dict 'sentences.db' \
-trace_dict 'trace.db' \
-annoy_index 'annoy_index.ann' \
-csv_file 'eventprimitives_pruned.csv' \
-words_dict 'words.db' \
-xq 'xqw.dat' \
-qsents 'qsentences.db' \
-qwords qwords.db \
> 'AWS_code/8out_5mo/word_sim_search.stdout' 2>&1 
'''


'''global argparser'''
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-batch_size', type=int, default=175, required=False, help='batch_size')
parser.add_argument('-annoy_index', type=str, required=False, help='annoy index filename')
parser.add_argument('-faiss_index', type=str, required=False, help='faiss index filename')
parser.add_argument('-csv_file', type=str, required=True, help='csv filename (results)')
parser.add_argument('-top_n', type=int, required=True, help='number of top n results you want to return for every query word')
parser.add_argument('-num_trees', type=int, required=True, help='num of trees for annoy index')
parser.add_argument('-dim', type=int, default=768, required=False, help='dimensions required to build annoy index')
parser.add_argument('-sentences_dict', type=str, required=True, help='sentence index filename')
parser.add_argument('-trace_dict', type=str, required=True, help='trace index filename')
parser.add_argument('-words_dict', type=str, required=True, help='word index filename')
parser.add_argument('-xq', type=str, required=True, help='query word embeds filename')
parser.add_argument('-qsents', type=str, required=True, help='query sentences index filename')
parser.add_argument('-qwords', type=str, required=True, help='query word index filename')
parser.add_argument('-outDir', type=str, required=True, help='Directory where all outfiles will be written to.')
parser.add_argument('--FAISS', action='store_false', dest='ANNOY_flag', required=False, help='Enable FAISS as the index')
parser.add_argument('--ANNOY', action='store_true', dest='ANNOY_flag', required=False, help='Enable ANNOY as the index')
args = parser.parse_args()

'''global variables'''
bert_tokenizer, bert_model = None, None
data = open(args.outDir+"shapes.txt","r")
shapes = data.read().splitlines()
shapes_dict = {}
for item in shapes: 
    fname,length = item.split(" ") 
    shapes_dict[fname]=length

def get_slice(it, size, section):
    '''I could change this to only return the start and end index of each subarray instead of all the indices
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))[section]                                       


                                #################################################
                                ##########      WORD SIM SEARCH         #########
                                #################################################


## Run query vectors using ANNOY similarity search
def word_sim_search(csv_fname, event, writer, xq_fname, qsents_fname, qwords_fname):
    ''' reloads files and runs queries and displays the results

    Args:
    Returns:
    Raises:
    '''
    global args, shapes_dict
    skipped, matches = 0, Counter()

    top_n = args.top_n

    if args.ANNOY_flag:

        print('running word queries using annoy...')

        ## Load data
        annoy_index = AnnoyIndex(args.dim, 'euclidean') 
        annoy_index.load(args.outDir+args.annoy_index)

        sentences = SqliteDict(args.outDir+args.sentences_dict)
        trace = SqliteDict(args.outDir+args.trace_dict)
        words = SqliteDict(args.outDir+args.words_dict)  ## tuple(word_id, word, (doc_id, sent_id)) for each content word
        
        xq = np.memmap(xq_fname, dtype='float32', mode='r', shape=(int(shapes_dict[xq_fname]),768)) ## word embeddings of the query matrix
        q_sentences = SqliteDict(qsents_fname) ## sqlite_dict of query sentences {sent_id: query_sentence}
        q_words = SqliteDict(qwords_fname)  ## sqlite_dict {sent_id: (sent_id, word_id, content_query_word)}
        matches = Counter()

        search_k = top_n * args.num_trees
        print('top_n requested: ', top_n)
        print('num_trees requested: ', args.num_trees)
        print('search_k = top_n * num_trees = %d' % (search_k))
        print('annoy_index fname: ', args.outDir+args.annoy_index)
        print('annoy_index.ntrees: ', annoy_index.get_n_trees())
        print('annoy_index.ntotal: ', annoy_index.get_n_items())
            
        ## Display top_n results based on querying the word
        for query_id, query_vec in enumerate(xq):
            prev_sent, match = '*', 1
            ## get query word, orig sent that corresponds to this query_vec
            ## format of val in sqlite_dict: (qword_id, qword, (doc_id, sent_id))
            ## format of qwords sqlite dict: {qword_id: (qword_id, qword, (doc_id, sent_id))}
            word_id, query_word, sent_id = q_words[query_id][0], q_words[query_id][1], q_words[query_id][2][1]
            # cur_freq = word_counts[query_word]
            knns = annoy_index.get_nns_by_vector(query_vec, top_n, search_k) ## cur_freq+top_n
            cur_query_sent = q_sentences[sent_id]
            print('\nQuery Word %d: (%s,%d)    len(knns): %d' % (query_id+1, query_word, word_id, len(knns)))
            print('\nOrigin: %s' % (cur_query_sent))
            print('\n*** Rankings ***')
            for nn_id, nearest_neighbor in enumerate(knns):
                ## (word_id, word, (doc_id, sent_id))
                doc_id, sent_id, word_id, cur_word, dist, sim_score = words[nearest_neighbor][2][0], words[nearest_neighbor][2][1], words[nearest_neighbor][0], words[nearest_neighbor][1], annoy_index.get_distance(query_id, nearest_neighbor), nlp_analysis.cosine_sim(query_vec, annoy_index.get_item_vector(nearest_neighbor))               
                # sentence_index = trace.index((doc_id, sent_id))
                cur_sent = sentences[sent_id]
                if cur_sent.startswith(prev_sent):
                    skipped += 1
                    continue
                elif sim_score < 0.60:
                    break
                elif matches[cur_word] > 4:
                    continue
                else:
                    print('\n%d\tNearest_Neighbor: (%s,%d)' % (match, cur_word, word_id))
                    print('\tdoc_id: %s  sent_id: %d  dist: %d cosine_sim: %f' % (doc_id, sent_id, dist, sim_score))
                    wrapper = textwrap.TextWrapper(subsequent_indent='\t')
                    print('\tOrigin: ', wrapper.fill(cur_sent))
                    writer.writerow({'event_primitive':event, 'query_sent':cur_query_sent, 'query_word':query_word, 'retrieved_word':cur_word, 'cosine_sim':sim_score, 'retrieved_sent':cur_sent,  'doc_id':doc_id, 'sent_id':sent_id, 'word_id':word_id})
                    match += 1
                    prev_sent = cur_sent[:15]
                matches.update([cur_word])
                

            print('\ntotal skipped sentences for this query word: ', skipped)
        print('\nnumber of unique words greater than cosine_sim 0.6:\t', len(matches.keys()))

    else:
        ## run queries using faiss
        print('running word queries using faiss...') 

        ## Load data
        faiss_index = faiss.read_index(args.outDir+args.faiss_index) 
        sentences = util.pickle_load(args.outDir+args.sentences_dict)
        trace = util.pickle_load(args.outDir+args.trace_dict)
        word_counts = util.pickle_load(args.outDir+args.word_counts)
        words = util.pickle_load(args.outDir+args.words_dict)  ## tuple(word_id, word, (doc_id, sent_id)) for each content word
        xq = util.pickle_load(xq_fname) ## word embeddings of the query matrix
        q_sentences = util.pickle_load(qsents_fname) ## list of query sentences
        q_words = util.pickle_load(qwords_fname)  ## tuple: (sent_id, word_id, content_query_word)
        matches = Counter()

        xq = np.vstack(xq).astype('float32')

        ## Run queries
        faiss.normalize_L2(xq)
        D, I = faiss_index.search(xq, 1200) # actual search
        skipped = 0

        print('\nKNN Matrix:\n', I)
        print('\nDistance Matrix:\n', D)
        print('\nRESULTS:')

        ## Display top_n results based on querying the word
        for query_id, query_vec in enumerate(I):
            prev_sent, match = '*', 1
            ## get query word, orig sent that corresponds to this query_vec
            # sent_id, word_id, query_word = q_words[query_id][0], q_words[query_id][1], q_words[query_id][2]
            word_id, query_word, sent_id = q_words[query_id][0], q_words[query_id][1], q_words[query_id][2][1]
            cur_freq = word_counts[query_word]
            cur_query_sent = q_sentences[sent_id]
            print('\nQuery Word %d: (%s,%d)    freq: %d    len(knns): %d' % (query_id+1, query_word, word_id, cur_freq, len(query_vec)))
            print('\nOrigin: %s' % (cur_query_sent))
            print('\n*** Rankings ***')
            ## tuples: (doc_id, paragraph_id, sent_id, word.strip())
            for nn_id, nearest_neighbor in enumerate(list(query_vec)):
                doc_id, sent_id, word_id, cur_word, sim_score = words[nearest_neighbor][2][0], words[nearest_neighbor][2][1], words[nearest_neighbor][0], words[nearest_neighbor][1], D[query_id][nn_id]               
                sentence_index = trace.index((doc_id, sent_id))
                cur_sent = sentences[sentence_index]
                if cur_sent.startswith(prev_sent):
                    skipped += 1
                    continue
                elif sim_score < 0.60:
                    break
                elif matches[cur_word] > 5:
                    continue
                else:
                    print('\n%d\tNearest_Neighbor: (%s,%d)' % (match, cur_word, word_id))
                    print('\tdoc_id: %s  sent_id: %d \t cosine_sim: %f' % (doc_id, sent_id, sim_score))
                    wrapper = textwrap.TextWrapper(subsequent_indent='\t')
                    print('\tOrigin: ', wrapper.fill(cur_sent))
                    writer.writerow({'event_primitive':event, 'query_sent':cur_query_sent, 'query_word':query_word, 'retrieved_word':cur_word, 'cosine_sim':sim_score, 'retrieved_sent':cur_sent,  'doc_id':doc_id, 'sent_id':sent_id, 'word_id':word_id})
                    match += 1
                    prev_sent = cur_sent[:10]
                    # print('this is a test')
                matches.update([cur_word])

            print('\ntotal skipped sentences for this query word: ', skipped)


if __name__ == '__main__':

    print('---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    csv_fname = args.outDir+args.csv_file

    start = time.time()
    ## write to csv file
    with open(csv_fname, mode='a') as csv_file:
        fieldnames = ['event_primitive', 'query_sent', 'query_word', 'retrieved_word', 'cosine_sim', 'retrieved_sent',  'doc_id', 'sent_id', 'word_id']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        ## xq files (query data) - numpy memmap and sqlite_dicts
        xq_fname = args.outDir+args.xq ##qword embeddings matrix
        qsents_fname = args.outDir+args.qsents
        qwords_fname = args.outDir+args.qwords

        print('running word_sim_search...')
        ## Run all the query vectors in the query matrix for sentence similarity
        word_sim_search(csv_fname, "query_sentences", writer, xq_fname, qsents_fname, qwords_fname)

    elapsed_time = time.time() - start
    print('Time elapsed running queries:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 

'''
## To run this file:
## Create virtual environment on nlpgrid
python3 -m venv <path_for_virtual_environment>

## Reno example:
python3 -m venv ~/venv3/ ##becca's venv: giga

## Activate virtual environment
source <path_for_virtual_environment>/bin/activate

## Reno example:
source ~/venv3/bin/activate 
e.g. ~/giga/bin/activate ##becca's venv: giga

## Install packages necessary
pip install nltk
pip install numpy
pip install tqdm
pip install torch==1.4.0
pip install transformers
pip install wheel
pip install annoy
pip install faiss-gpu (see section on installing faiss for more info)
pip install sklearn
pip install -U sentence-transformers
pip install ndjson
python3 -m spacy download en_core_web_lg

## confirm torch version
python3
>>>import torch
>>>print(torch.__version__) //should be 1.4.0

## installing faiss
    to check which cuda version you have in nlpgrid
        cat /usr/local/cuda/version.txt
    for CPU version
        conda install faiss-cpu -c pytorch
    for GPU version
        conda install faiss-cpu -c pytorch
        conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
        conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
        conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10
    for nlpgrid gpus
        pip install faiss-gpu
## confirm faiss
    python3
    >>>import faiss
    >>>import numpy as np
## confirm annoy
    python3
    >>>import annoy
    >>>from annoy import AnnoyIndex
    
'''
