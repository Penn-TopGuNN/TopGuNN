import sys, csv 
from tqdm import tqdm
import util
from collections import defaultdict, Counter
import time
from itertools import islice
import functools
import pdb
import os
import argparse
import nlp_analysis
import annoy
from annoy import AnnoyIndex
import textwrap
from textwrap import indent
import time
import numpy as np
from sqlitedict import SqliteDict
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
from itertools import chain
import timeit
import json
import synchronized_map
from threading import Lock

'''usage:
source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N synchronized_word_sim_search
#$ -l h=nlpgrid10
#$ -l h_vmem=150G
python3 -u code/synchronized_word_sim_search.py \
-num_workers 5 \
-queryDir 'betatest/data/' \
-outDir 'betatest/out/' \
-annoy_run_id 'annoy_index' \
-top_n 1000 \
-top_n_increment 500 \
-top_n_threshold 3000 \
-num_uniq 10 \
-acc 1 \
--SINGLE_QUERY \
> 'betatest/out/synchronized_word_sim_search.stdout' 2>&1 
'''

'''global argparser'''
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-num_workers', type=int, default=4, required=True, help='nworkers for multi-processing')
parser.add_argument('-outDir', type=str, required=True, help='Directory where all outfiles will be written to.')
parser.add_argument('-queryDir', required=True, help='where to read the event names from.')
parser.add_argument('-annoy_run_id', type=str, required=True, help='How will the annoy indexes be named (the prefix)?')
parser.add_argument('-batch_size', type=int, default=175, required=False, help='batch_size')
parser.add_argument('-top_n', type=int, required=True, help='number of top n results you want to return for every query word')
parser.add_argument('-top_n_increment', type=int, required=False, default=500, help='how much top_n should increase by for the next re-query if not enough unique results')
parser.add_argument('-top_n_threshold', type=int, required=False, default=3000, help='number of top n results you want to return for every query word')
parser.add_argument('-num_uniq', type=int, required=False, default=10, help='number of unique results required')
parser.add_argument('-acc', type=float, required=False, default=1.0, help='The accuracy from 0 to 1')
parser.add_argument('-dim', type=int, default=768, required=False, help='dimensions required to build annoy index')
parser.add_argument('--SINGLE_QUERY', action='store_false', dest='MULTI_INDICATED', required=False, help='Indicates there is only 1 query matrix.')
parser.add_argument('--MULTI_QUERY', action='store_true', dest='MULTI_INDICATED', required=False, help='Indicates if there are several query matrices.')
args = parser.parse_args()


'''global variables'''
print('\n---shapes dict---')
bert_tokenizer, bert_model = None, None
data = open(args.outDir+"shapes.txt","r")
shapes = data.read().splitlines()
data.close()
shapes_dict = {}
for item in shapes:
#    print(item)
    fname,length = item.split(" ") 
    shapes_dict[fname]= int(length)

## words_fnames
words_fnames = sorted([join(args.outDir+f) for f in listdir(args.outDir) if f.startswith('words_job')])
#print('\nwords_fnames: ', len(words_fnames))
#for fname in words_fnames:
#    print(fname)

## total sums of lengths for each partition
total_sum_lengths_dict,prev = {},0
total_sum_lengths_dict[0] = 0
for i,fname in enumerate(words_fnames[1:],start=1):
    total_sum_lengths_dict[i] = shapes_dict[words_fnames[i-1]]+prev
    prev = total_sum_lengths_dict[i]
#print(f'\ntotal_sum_lenghts_dict: {total_sum_lengths_dict}')
## total_sum_lenghts_dict: {0: 0, 1: 1883930, 2: 3829120, 3: 5732939, 4: 7680058}


print('\n---annoy indexes---')
annoy_indexes_fnames = sorted([join(args.outDir,f) for f in listdir(args.outDir) if f.startswith(args.annoy_run_id) and f.endswith('.ann')])
#print(f'len(annoy_indexes_fnames): {len(annoy_indexes_fnames)}\nannoy_index_fnames: ')
#for fname in annoy_indexes_fnames:
#    print(fname)

## Load Annoy Indexing System
annoy_indexes = [AnnoyIndex(args.dim, 'euclidean') for i in range(len(annoy_indexes_fnames))] #*len(annoy_indexes_fnames)[:1]
for annoy_index, fname in zip(annoy_indexes, annoy_indexes_fnames):
    annoy_index.load(fname) 

annoy_time = 0

event_names = None
if args.MULTI_INDICATED:
    ## event names
    print('\n---event names---')
    event_names = [f[:-5] for f in listdir(args.queryDir) if f.endswith('.json')]
    print(f'len(event_names): {len(event_names)}\nevent names: ')
    for event in event_names:
        print(event)
else:
    event_names = ['']

print("\nevent_names: ", event_names)






                                #################################################
                                ##########      HELPER FUNCTIONS        #########
                                #################################################

display_and_write_results_lock = Lock()
def display_and_write_results(event, query_to_display, retreived_to_display, writer):

    print(f'\n\n\n---RESULTS FOR: {event}---')
    print ('len(query_to_display) {}, len(retreived_to_display) {}'.format(len(query_to_display), len(retreived_to_display)))

    for idx, quer_res in enumerate(query_to_display):

        ## tuple: (query_id+1, query_word, qword_id, len(knns), cur_query_sent)
        query_id, query_word, qword_id, knns, cur_query_sent = quer_res[0], quer_res[1], quer_res[2], quer_res[3], quer_res[4]
        print('\nQuery Word %d: (%s,%d)    len(knns): %d' % (query_id, query_word, qword_id, knns))
        print('\nOrigin: %s' % (cur_query_sent))
        print('\n*** Rankings ***')

        for cur_tuple in retreived_to_display[query_id]['results']:

            ## {results: tuple(all this info), skipped: 6}
            match, cur_word, word_id, nearest_neighbor, doc_id, sent_id, dist, sim_score, cur_retrieved_sent, event = cur_tuple[0], cur_tuple[1], cur_tuple[2], cur_tuple[3], cur_tuple[4], cur_tuple[5], cur_tuple[6], cur_tuple[7], cur_tuple[8], cur_tuple[9]
            print(f'\n{match}\tNearest_Neighbor: {cur_word, word_id} knn#: {nearest_neighbor}')
            print(f'\tdoc_id: {doc_id}  sent_id: {sent_id}  dist: {dist} cosine_sim: {sim_score}')
            wrapper = textwrap.TextWrapper(subsequent_indent='\t')
            print('\tOrigin: ', wrapper.fill(cur_retrieved_sent))
            writer.writerow({'event_primitive':event, 'query_sent':cur_query_sent, 'query_word':query_word, 'retrieved_word':cur_word, 'cosine_sim':sim_score, 'retrieved_sent':cur_retrieved_sent,  'doc_id':doc_id, 'sent_id':sent_id, 'word_id':word_id})

        print('\ntotal skipped sentences for [{}]: {} '.format(query_word, retreived_to_display[idx]['skipped']))
    print(f'---END RESULTS FOR: {event}---\n\n\n')

def get_slice(it, size, section):
    '''I could change this to only return the start and end index of each subarray instead of all the indices
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))[section]                                       


                                #################################################
                                ##########      WORD SIM SEARCH         #########
                                #################################################

def get_amalgamated_knn(knn, index_of_the_annoy_index):
    return total_sum_lengths_dict[index_of_the_annoy_index] + knn

def job_func(job_args):
    if job_args is None:
        return None
    index_of_the_annoy_index, query_vec, query_id, query_word, top_n = job_args 
    annoy_index = annoy_indexes[index_of_the_annoy_index]
    search_k = int(args.acc * top_n * annoy_index.get_n_trees())
    print(f'Searching Annoy Index: {index_of_the_annoy_index} for query word: {query_word} with query id: {query_id}')
    # print(f'---job args--- \nindex_of_the_annoy_index: {index_of_the_annoy_index} \ntop_n: {top_n} \nsearch_k: {search_k}')
    # print('line 140 starting get_nns_by_vector')
    # print(f'type(annoy_index): {type(annoy_index)}')
    # print(f'type(query_vec): {type(query_vec)} len(query_vec): {len(query_vec)}')
    knns, knn_distances = annoy_index.get_nns_by_vector(query_vec, top_n, search_k, include_distances = True)
    amalgamated_knns = [get_amalgamated_knn(knn, index_of_the_annoy_index) for knn in knns]
    # print(f'\n---annoy_index {index_of_the_annoy_index}---')
    # for local_knn,global_knn in zip(knns[:10],amalgamated_knns[:10]):
    #     print(f'old knn_id: {local_knn} global_knn_id: {global_knn}')
    return amalgamated_knns, knn_distances

def parallel_get_nns_by_vector(query_id, event_idx, query_vec, query_word, top_n):
    # run the query against all annoy indexes in parallel using multiprocessing Pool  
    global args
    top_ns = [top_n]*len(annoy_indexes)
    query_vecs = [query_vec]*len(annoy_indexes)
    results_from_each_annoy_index = synchronized_map.map((event_idx, query_id), job_func, list(zip(list(range(len(annoy_indexes))), query_vecs, [query_id]*len(annoy_indexes), [query_word]*len(annoy_indexes), top_ns)))
  
    k_temp, d_temp = zip(*results_from_each_annoy_index)
    knns, distances = list(chain.from_iterable(k_temp)), list(chain.from_iterable(d_temp))
    sorted_top_results = sorted(zip(distances, knns))[:top_n]
    sorted_distances, sorted_knns = list(zip(*sorted_top_results))
  
    return sorted_knns, sorted_distances


## Run query vectors using ANNOY similarity search
def word_sim_search(func_args):
    ''' reloads files and runs queries and displays the results

    Args:
    Returns:
    Raises:
    '''
    global args, shapes_dict, annoy_time

    event_idx, other_args = func_args
    event, sentences, trace, words, csv_fname, xq_fname, qsents_fname, qwords_fname = other_args

    #Load Data
    xq = np.memmap(xq_fname, dtype='float32', mode='r', shape=(int(shapes_dict[xq_fname]),args.dim)) ## word embeddings of the query matrix
    q_sentences = SqliteDict(qsents_fname) ## sqlite_dict of query sentences {sent_id: query_sentence}
    q_words = SqliteDict(qwords_fname)  ## sqlite_dict {sent_id: (sent_id, word_id, content_query_word)}
    query_word_results, retrieved_word_results = [], defaultdict(lambda: defaultdict(lambda: []))

    @functools.lru_cache(maxsize=50000)
    def look_up_sentence(sent_id):
        return sentences[sent_id]

    @functools.lru_cache(maxsize=len(xq)*50000)
    def look_up_word(word_id):
        return words[word_id]

    def num_unique_words(knns, counter, increment):
        for nearest_neighbor in knns[-increment:]:
            ## {word_id: (word_id, word, (doc_id, sent_id))}
            counter.update([look_up_word(nearest_neighbor)[1].lower()])
        return len(counter)

    # Helper to run over a single word
    def word_sim_search_single_word(func_args):
        global annoy_time
        
        _, other_args = func_args
        query_id, query_vec = other_args

        top_n = args.top_n
        counter = Counter()
        skipped=0

        requery_number = 0
        while True: # requery loop
            prev_sent, match = '*', 1
            ## tuple {qword_id: (qword_id, qword, (doc_id, sent_id))}
            qword_id, query_word, sent_id = q_words[query_id][0], q_words[query_id][1], q_words[query_id][2][1]
            print('\nquery_word {}: {}'.format(query_id+1, query_word))
            annoy_start = time.time()
            knns, knn_distances = parallel_get_nns_by_vector(query_id, event_idx, query_vec, query_word, top_n)
            annoy_end = time.time() - annoy_start
            annoy_time += annoy_end
            num_unique = num_unique_words(knns, counter, i)
            if top_n == args.top_n_threshold and num_unique-1 < args.num_uniq:
                ## if 10 unique words aren't ikippeesynchronized_map.maprd: {}'.format(query_word))
                print(f'\nDid not find enough num of unique for: --{query_word}--! current num of unique: {num_unique}\n')
                synchronized_map.mark_as_done((event_idx, query_id))
                break
            elif num_unique-1 < args.num_uniq: ## minus 1, b/c we assume exact string match will be included in the num_unique and we don't want to count it
                ## iteratively increase top_n by 100 until 10 unique words are in its knns
                top_n += args.top_n_increment
                requery_number += 1
                print('\nNeed to re-query query word: {} on re-query iteration: {}'.format(query_word, requery_number))
                synchronized_map.mark_as_not_done((event_idx, query_id))
                continue
            else:  
                print('\nFound {} unique results for query word: {} with query id: {} using top-N: {}'.format(num_unique, query_word, query_id, top_n))
                ## save query results to be displayed later
                cur_query_sent = q_sentences[sent_id]
                query_word_results.append((query_id+1, query_word, qword_id, len(knns), cur_query_sent))

                ## run queries per the usual
                matches = Counter()
                for nn_id, nearest_neighbor in enumerate(knns):
                    ## (word_id, word, (doc_id, sent_id))
                    words_nn = look_up_word(nearest_neighbor)
                    doc_id, sent_id, word_id, cur_word, dist, sim_score = words_nn[2][0], words_nn[2][1], words_nn[0], words_nn[1], round(knn_distances[nn_id],3), round((2-(knn_distances[nn_id]**2))/2,3) #annoy_index.get_distance(query_id, nearest_neighbor), nlp_analysis.cosine_sim(query_vec, annoy_index.get_item_vector(nearest_neighbor))                                        
                    cur_retrieved_sent = look_up_sentence(sent_id)                       
                    if cur_retrieved_sent.startswith(prev_sent):
                        skipped += 1
                        continue
                    elif query_word.lower() == cur_word.lower():
                        continue
                    elif sim_score < 0.60:
                        break
                    elif matches[cur_word] > 4:
                        continue
                    else:
                        retrieved_word_results[query_id+1]['results'].append((match, cur_word, word_id, nearest_neighbor, doc_id, sent_id, dist, sim_score, cur_retrieved_sent, event))
                        match += 1
                        prev_sent = cur_retrieved_sent[:40]
                    matches.update([cur_word])
                retrieved_word_results[query_id+1]['skipped'].append(skipped)
                synchronized_map.mark_as_done((event_idx, query_id))
                break

    ## Display top_n results based on querying the word
    synchronized_map.synchronize_over(word_sim_search_single_word, [(
            query_id,
            query_vec,
        ) for query_id, query_vec in enumerate(xq)])


    ## Cleanup
    del xq
    q_sentences.close()
    q_words.close()


    ## Write to CSV and print results
    display_and_write_results_lock.acquire() # Prevent two prints of the results from getting inter-leaved
    with open(csv_fname, mode='w+') as csv_file_run_id:
        fieldnames = ['event_primitive', 'query_sent', 'query_word', 'retrieved_word', 'cosine_sim', 'retrieved_sent',  'doc_id', 'sent_id', 'word_id']
        writer = csv.DictWriter(csv_file_run_id, fieldnames=fieldnames)
        writer.writeheader()
        display_and_write_results(event, query_word_results, retrieved_word_results, writer)
    display_and_write_results_lock.release() 


if __name__ == '__main__':

    start = time.time()

    print('\n---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    total_query_words = 0
    if args.MULTI_INDICATED:

        print("multiple query matrices indicated!")

        # multiple query matrices indicated (mutliple events to run queries for)
        # Make sure all files exist or error immediately if they don't
        for event_name in event_names:
            assert os.path.exists(args.queryDir+'xq_'+event_name+'.dat')
            assert os.path.exists(args.queryDir+'qsentences_'+event_name+'.db')
            assert os.path.exists(args.queryDir+'qwords_'+event_name+'.db')
      
        for event_name in event_names:
            xq_fname = args.queryDir+'xq_'+event_name+'.dat'
            xq = np.memmap(xq_fname, dtype='float32', mode='r', shape=(int(shapes_dict[xq_fname]),args.dim)) ## word embeddings of the query matrix
            total_query_words += len(xq)
            del xq

        print("Total Query Words:", total_query_words)

        # Load data
        sentences = SqliteDict(args.outDir+'sentences.db')
        trace = SqliteDict(args.outDir+'trace.db')
        words = SqliteDict(args.outDir+'words.db')  ## tuple(word_id, word, (doc_id, sent_id)) for each content word       


        # Run word_sim search over all events
        synchronized_map.init(args.num_workers, total_query_words, _log = True)
        synchronized_map.synchronize_over(word_sim_search, [(
                event_name,
                sentences,
                trace,
                words,
                args.queryDir+event_name+'.csv',
                args.queryDir+'xq_'+event_name+'.dat',
                args.queryDir+'qsentences_'+event_name+'.db',
                args.queryDir+'qwords_'+event_name+'.db',
            ) for event_name in event_names])

    else:

        print("single query matrix indicated!") 

        # there is only a single query matrix (1 event)
        assert os.path.exists(args.queryDir+'xq.dat')
        assert os.path.exists(args.queryDir+'qsentences.db')
        assert os.path.exists(args.queryDir+'qwords.db')

        xq_fname = args.queryDir+'xq.dat'
        xq = np.memmap(xq_fname, dtype='float32', mode='r', shape=(int(shapes_dict[xq_fname]),args.dim)) ## word embeddings of the query matrix
        total_query_words += len(xq)
        del xq

        print("Total Query Words:", total_query_words)

        # Load data
        sentences = SqliteDict(args.outDir+'sentences.db')
        trace = SqliteDict(args.outDir+'trace.db')
        words = SqliteDict(args.outDir+'words.db')  ## tuple(word_id, word, (doc_id, sent_id)) for each content word       


        # Run word_sim search over single event
        synchronized_map.init(args.num_workers, total_query_words, _log = True)
        synchronized_map.synchronize_over(word_sim_search, [(
                event_name,
                sentences,
                trace,
                words,
                args.queryDir+'synchronized_word_sim_search_results.csv',
                args.queryDir+'xq'+event_name+'.dat',
                args.queryDir+'qsentences'+event_name+'.db',
                args.queryDir+'qwords'+event_name+'.db',
            ) for event_name in event_names])

    # Cleanup
    synchronized_map.close()
    sentences.close()
    trace.close()
    words.close()

    elapsed_time = time.time() - start

    print('Time elapsed running queries:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    print('Time elapsed for annoy time:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(annoy_time/total_query_words))))


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

In [64]: should_requery=True                                                                                                                

In [65]: topn=10                                                                                                                            

In [66]: names                                                                                                                              
Out[66]: ['chris', 'becca', 'ajay', 'daphne', 'reno']

custom_start = 0 
while should_requery: 
    # pdb.set_trace() 
    should_requery = False 
    for i, name in enumerate(names[custom_start:], start=custom_start): 
        print('being processed: ', name) 
        if len(name) <= topn: ## this in practice would be len(unique_words) 
            topn -= 1 ## in practice this will be some condition I need to adjust 
            should_requery = True 
            break 
        else: 
            ## if you've reached the end of the loop then exit 
            if i == len(names)-1: 
                should_requery = False 
            else: 
                topn = 10 
                custom_start += 1 
                ## this is wear the normal query code would go

mp over annoy index (only 5)
disable all queries (just do one word)
query each index for 50 words and combine results (sort them internally in python)
take top 50 from the 250 and then that write to CV.
consine_sim = (2-(dist ** 2))/2
    
'''
