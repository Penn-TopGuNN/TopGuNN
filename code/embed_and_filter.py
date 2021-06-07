import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader 
import util
from util import MaskableList
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import spacy
import time
import itertools
from itertools import islice
import os
import argparse
from sklearn.preprocessing import normalize
from sqlitedict import SqliteDict
import ast
import pickle as pkl
import sqlite3
## TODO: line 244, 264 - determine increment at which to report stats for stdout (CTRL-F) 'if round_id % 100 == 0:' to check if in script
## TODO: remove '__debugging_delete_after' keyword from debuggin (CTRL-F) to check if in script
## TODO: Remove debugging lines - 185-187
## TODO: commented out line 124, b/c we have pre-clipped spacy toks no need to clip bert_NER_toks inside the if statement
## TODO: update correct file on line 508
## TODO: CTRL-F "finalcheck" or "mpcheck" remove debugging statements from filenames

nlp = spacy.load("en_core_web_lg", disable=["ner"]) ## you only need the parser and tagger
## device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ##.to(device)
## NOTE: once debugging is ironed-out remove all print statements, csv file, and time study files, for AWS

'''usage: (goes in shell script)
python3 -u AWS_code/embed_and_filter.py \
-job_id $i \
-outDir 'AWS_code/bigjob/' \
-dataDir 'data/' \
-NUM_JOBS 2 \
-NUM_GPUS 2 \
-PROC_PER_GPU 1 \
-gpu_ids 67 \
-batch_size 175 \
-clip_len 225 \
-job_slices "job_slices.pkl" \
-query_sentences 'data/query_sentences.txt' \
-sentences_dict 'sentences.db' \
-trace_dict 'trace.db' \
-spacy_toks_dict 'spacy_toks.db' \
-spacy_pos_dict 'spacy_pos.db' \
-spacy_deps_dict 'spacy_deps.db' \
--BERT \
--MEAN \
> 'AWS_code/bigjob/embed_and_filter_job'$i'.stdout' 2>&1
alternative: | tee beta_testing/job_array/stdout_job_array.txt) 3>&1 1>&2 2>&3 | tee beta_testing/job_array/stderr_job_array.txt
'''

'''global argparser'''
total_nword_embeddings, nskipped, time_elapsed_embedding, time_elapsed_filtering = 0, 0, 0, 0
bert_tokenizer, bert_model = None, None
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help='Directory where all outfiles will be written to. Example: out/')
parser.add_argument('-dataDir', required=True, help='Directory where all data files are located. Example: data/')
parser.add_argument('-job_id', required=True, help='job_id responsible for x-partition of the amalgams.')
# parser.add_argument('-NUM_JOBS', type=int, required=True, help='example: 5 (should match npartitions==NUM_GPUS)')
parser.add_argument('-batch_size', type=int, required=True, help='example: 400 (400 sentences in each batch)')
parser.add_argument('-clip_len', type=int, required=True, help='number of sentences to batch')
#parser.add_argument('-NUM_GPUS', type=int, required=True, help='number of GPUs')
#parser.add_argument('-PROC_PER_GPU', type=int, required=True, help='number of processes per GPU')
parser.add_argument('-gpu_id', type=int, required=True, help='list gpu_ids available separated by white space, i.e. - 3 4 5 16')
parser.add_argument('-job_slices', type=str, required=True, help="the job slices file output from create_amalgams.py. Example: 'job_slices.pkl'")
parser.add_argument('-query_sentences', type=str, required=True, help="query sentences filename. Example: 'query_sentences.txt'")
parser.add_argument('-sentences_dict', required=True, help="sqlite db filename. Example: 'sentences_dict.db'")
parser.add_argument('-trace_dict', required=True, help="sqlite db filename. Example: 'trace_dict.db'")
parser.add_argument('-spacy_toks_dict', required=True, help="sqlite db filename. Example: 'spacy_toks_dict.db'")
parser.add_argument('-spacy_pos_dict', required=True, help="sqlite db filename. Example: 'spacy_pos_dict.db'")
parser.add_argument('-spacy_deps_dict', required=True, help="sqlite db filename. Example: 'spacy_deps_dict.db'")
parser.add_argument('--BERT', action='store_false', dest='SBERT_flag', required=False, help='Enable BERT as the model')
parser.add_argument('--MEAN', action='store_false', dest='HEAD_flag', required=False, help='Calculates embeddings using the mean of the subword units')
parser.add_argument('--SBERT', action='store_true', dest='SBERT_flag', required=False, help='Enable SBERT as the model')
parser.add_argument('--HEAD', action='store_true', dest='HEAD_flag', required=False, help='Calculates embedding using only the headword embedding of the subword unit')
args = parser.parse_args()

'''global variables'''
## load job partition file
job_slices = util.pickle_load(args.outDir+args.job_slices)
print('\nlen(job_slices): {}'.format(len(job_slices)))


                                #################################################
                                ##########     Embed and Filter      ############
                                #################################################

                    
def embed_sentences(round_id, sentence_batch, trace_batch, spacy_toks, spacy_pos, spacy_deps): ## , bert_tokenizer, bert_model, SBERT_flag, HEAD_flag
    ''' Takes in a batch of sentences and generates BERT embeddings for them. 
    Args:
    Returns:
    Note:
        remove bert_tokenizer, bert_model, SBERT_flag, HEAD_flag from method signature when not running multiprocessing
        make sure SBERT_flag, and HEAD_flag are added back in
    '''
    global time_elapsed_embedding, time_elapsed_filtering
    global bert_tokenizer, bert_model, args
    start_embed_time = time.time()

    cur_words, cur_embeds = [], []
    content_tags = ['ADJ', 'ADV', 'NOUN', 'VERB']
    aux_tags = ['aux', 'auxpass', 'poss', 'possessive', 'cop', 'punct']

    ## tensor board, web ui (pytorch)
    ## perform lowercasing of all the sentences for embeddings
    sent_iter=iter(sentence_batch)
    lowercased_sentence_batch = [sent.lower() for sent in sent_iter]
    
    if args.SBERT_flag:

        return bert_model.encode([sentence])[0]

    else:

        ##pytorch logging library

        # try:

        ## batched encoding is a dict with keys = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        NER_encoded_batch = [bert_tokenizer.batch_encode_plus(tok) for tok in spacy_toks] ## bert_NER_toks 
        # encoded_batch = bert_tokenizer.batch_encode_plus(lowercased_sentence_batch) ## regular bert_toks    
        
        ## We want BERT to process our examples all at once (as one lowercased_sentence_batch).
        ## For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array.
        padded_batch = bert_tokenizer.batch_encode_plus(lowercased_sentence_batch, pad_to_max_length=True)

        ## Grab indices and attn masks from the padded lowercased_sentence_batch.
        ## We need to tell BERT to ignore (mask) the padding we've added when it's processed as input.
        padded_input_ids, attention_masks = np.array(padded_batch['input_ids']), np.array(padded_batch['attention_mask'])

        NER_iter = iter(NER_encoded_batch)
        bert_NER_toks = [[bert_tokenizer.convert_ids_to_tokens(NER_unit)[1:-1] for NER_unit in cur_dict['input_ids']] for cur_dict in NER_iter]

        padded_tinput_ids = torch.tensor(padded_input_ids).cuda() ##batched padded_input_ids converted to torch tensors
        attention_masks = torch.tensor(attention_masks).cuda()   ##batched attention_masks converted to torch tensors
        
        # print('padded_tinput_ids.size()[1] ', padded_tinput_ids.size())

        if padded_tinput_ids.size()[1] > args.clip_len:
            print('\n\nclipping sentences round {} '.format(round_id))
            # print('\nclipped sentences: ', sentence_batch)
            # print('\nbert_NER_toks: ', bert_NER_toks)
            # print(' after change round {} - type(padded_tinput_ids) and size: {}  {} '.format(i, type(padded_tinput_ids), padded_tinput_ids.size()))
            # bert_NER_toks = [NER_unit[:args.clip_len] for NER_unit in bert_NER_toks]
            print('before padded_tinput_ids.size: ', padded_tinput_ids.size())
            padded_batch = bert_tokenizer.batch_encode_plus(lowercased_sentence_batch, max_length=args.clip_len, pad_to_max_length=True)
            padded_input_ids, attention_masks = np.array(padded_batch['input_ids']), np.array(padded_batch['attention_mask'])
            print('padded_input_ids.dtype, attention_masks.dtype: ', padded_input_ids.dtype, attention_masks.dtype)
            padded_tinput_ids = torch.tensor(padded_input_ids).cuda() ##batched padded_input_ids converted to torch tensors
            attention_masks = torch.tensor(attention_masks).cuda()   ##batched attention_masks converted to torch tensors
            print('after padded_tinput_ids.size: ', padded_tinput_ids.size())
            print('---end clipped sentences---')
            print('\n\n')

        # print('after having been clipped - padded_tinput_ids.size: ', padded_tinput_ids.size())
        try:
            with torch.no_grad():
                embeds = bert_model(padded_tinput_ids, attention_mask=attention_masks)
        except RuntimeError:
            print('\n\nLine 143 CUDA out of memory. ')
            print('padded_tinput_ids.size: ', padded_tinput_ids.size())
            return -1

        ## Saves relevant word embeddings from the padding (removing [CLS] and [SEP] tokens)
        ## for each sentence, where the last token resides
        mask_iter = iter(np.array(attention_masks.cpu()))
        relevant_ids = np.array([[i,len(arr)-1-list(arr[::-1]).index(1)] for i, arr in enumerate(mask_iter)])
        ## changes [SEP] tokens attention to 0
        attention_masks[relevant_ids[:,0], relevant_ids[:,1]]=0 ## temp[:,0] return 0th col for all rows, temp[:,1]] return 1st col for all rows. Change corresponding [row, col] in arrays to 0
        ## changes [CLS] tokens attention to 0
        attention_masks[:,0]=0

        ## attention masks to be applied to relevant embeddings within each torch tensor
        mask_iter, embeds_iter = iter(attention_masks), iter(embeds[0]) 
        relevant_embeds = [MaskableList(sublist)[submask] for sublist, submask in zip(embeds_iter, mask_iter)]

        ## reflects the bert_NER full-token words (not bert's subword units)
        pos_iter, dep_iter = iter(spacy_pos), iter(spacy_deps)
        relevant_annotations_mask = [(np.in1d(cur_pos,content_tags)) & (~np.in1d(cur_dep,aux_tags)) for cur_pos, cur_dep in zip(pos_iter,dep_iter)]

        embed_time = time.time() - start_embed_time
        time_elapsed_embedding += embed_time

        start_filter_time = time.time()

        if args.HEAD_flag:
        ## use only embedding of the full-token word for each subword unit

            for i in range(len(bert_NER_toks)):
                end_index,j,k=0,0,0
                while(j<len(relevant_embeds[i])):
                    end_index=end_index+len(bert_NER_toks[i][k])
                    if relevant_annotations_mask[i][k]:
                        cur_words.append((k,spacy_toks[i][k],(trace_batch[i][0], int(trace_batch[i][1]))))
                        ## stack, mean, and numpy 'em
                        temp = torch.mean(torch.stack(relevant_embeds[i][j:j+1]),0).cpu().numpy()
                        cur_embeds.append(temp)
                    j,k=end_index,k+1        

        else:
        # use mean of subwords units to calculate embeddings
            try:          
                for i in range(len(bert_NER_toks)):
                    end_index,j,k=0,0,0                       
                    while(j<len(relevant_embeds[i])):
                        end_index=end_index+len(bert_NER_toks[i][k])
                        # if (round_id > 799 and round_id < 803) or (round_id > 984 and round_id < 988):
                        #     print('i {}, k {}, len(bert_NER_toks[i]) {},   bert_NER_toks[i][k] {}'.format(i, k, len(bert_NER_toks[i]), bert_NER_toks[i][k]))
                        #     print('bert_NER_toks[i]: ', bert_NER_toks[i])
                        if relevant_annotations_mask[i][k]:
                            cur_words.append((k,spacy_toks[i][k],(trace_batch[i][0], int(trace_batch[i][1]))))
                            ## stack, mean, and numpy 'em
                            temp = torch.mean(torch.stack(relevant_embeds[i][j:end_index]),0).cpu().numpy() ##is this end_index or end_index+1
                            cur_embeds.append(temp)
                        j,k=end_index,k+1 
            except IndexError as e:
                print('\n\n---IndexError: list index out of range!---')
                print(e)
                print('round_id: ', round_id)
                print('i, k:', i, k)
                print('len(sentence_batch), len(trace_batch[0]): ', len(sentence_batch), len(trace_batch[0]))
                print('len(bert_NER_toks)', len(bert_NER_toks))
                print('len(bert_NER_toks[i]): ', len(bert_NER_toks[i]))
                # print('\nbert_NER_toks[i]: ', bert_NER_toks[i])
                # print('\nbert_NER_toks', bert_NER_toks)
                print('--end current error--\n\n')

        filter_time = (time.time() - start_filter_time)
        time_elapsed_filtering += filter_time

            # print('round %d   Time elapsed filtering content words:\t%s' % (round_id, time.strftime("%H:%M:%S", time.gmtime(filter_time))))
        # except AttributeError:
        #     print('\n\n---AttributeError----NoneType object has no attribute batch_encode_plus!')
        #     print('spacy_toks: ')
        #     print(spacy_toks)
        #     print('trace_batch: ')
        #     print(trace_batch)
        #     print('sentence_batch: ')
        #     print(sentence_batch)
        #     print('print(list(sentence_batch)):')
        #     print(list(sentence_batch))
        #     print('---end of line---\n\n')

    if round_id % 100 == 0:
        print('finished batch {}. len(words): {} len(embeds): {}'.format(round_id, len(cur_words), len(cur_embeds)))

    return cur_words, cur_embeds

def embed_all_batches(batched_sentences, batched_trace_info, batched_spacy_toks, batched_spacy_pos, batched_spacy_deps):
    '''Iterates through giga_dict and batches sentences to send of embed_all_sentences().
    Args:
    Returns:
    Note:
    '''

    global args, total_nword_embeddings

    words, word_embeds = [], []

    batch_iter, trace_iter, spacy_toks_iter, spacy_pos_iter, spacy_deps_iter = iter(batched_sentences), iter(batched_trace_info), iter(batched_spacy_toks), iter(batched_spacy_pos), iter(batched_spacy_deps)

    for round_id, (sentence_batch, trace_batch, spacy_toks_batch, spacy_pos_batch, spacy_deps_batch) in enumerate(zip(batch_iter, trace_iter, spacy_toks_iter, spacy_pos_iter, spacy_deps_iter)):
  
        if round_id % 100 == 0:
            print('\nprocessing embedding {}... percentage processed {}'.format(round_id, (round_id/len(batched_sentences))*100))

        cur_words, cur_embeds = embed_sentences(round_id, sentence_batch, trace_batch, spacy_toks_batch, spacy_pos_batch, spacy_deps_batch) ## each batch is of size batch_size (see global var)

        words.extend(cur_words)
        word_embeds.extend(cur_embeds)

        total_nword_embeddings += len(cur_embeds)

    return words, word_embeds

                   
def handle_batches(cur_sentences, cur_trace, cur_spacy_toks, cur_spacy_pos, cur_spacy_deps, words_dict, word_embeds_fname):

    global args, job_slices, time_elapsed_embedding, time_elapsed_filtering

    embed_time, filtering_time = 0, 0
    batch_size, outDir = args.batch_size, args.outDir
    print('size of batch: ', batch_size)

    ## Reads in gigaword file
    # sentences, trace = read_file(gigaword_fname)
    print('len(sentences), len(trace), len(cur_spacy_toks), len(cur_spacy_pos), len(cur_spacy_deps): ', len(cur_sentences), len(cur_trace), len(cur_spacy_toks), len(cur_spacy_pos), len(cur_spacy_deps))

    ## use pytorch library DataLoader to batch sentences and nlp annotations
    batched_sentences = DataLoader(cur_sentences, batch_size=batch_size)
    batched_trace_info = DataLoader(cur_trace, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_toks = DataLoader(cur_spacy_toks, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_pos = DataLoader(cur_spacy_pos, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_deps = DataLoader(cur_spacy_deps, batch_size=batch_size, collate_fn=custom_collate)  
    
    print('DataLoader (batch_size %d): %d %d %d %d %d' %(batch_size, len(batched_sentences), len(batched_trace_info), len(batched_spacy_toks), len(batched_spacy_pos), len(batched_spacy_deps)))

    ## Embeds sentences from all batches
    words, word_embeds = embed_all_batches(batched_sentences, batched_trace_info, batched_spacy_toks, batched_spacy_pos, batched_spacy_deps) 

    print('these lengths should match:  len(words): {}, len(word_embeds): {}, total_nword_embeds_check: {} '.format(len(words), len(word_embeds), total_nword_embeddings))

    word_dict_start = time.time()
    words_iter = iter(words)
    idx_iter = range(len(words))
    words_dict.update([(idx,word) for idx,word in zip(idx_iter,words_iter)])
    words_dict.commit()
    words_dict.close()
    word_dict_time = time.time() - word_dict_start

    ## memmap word_embeds
    memmap_start = time.time()
    fp = np.memmap(word_embeds_fname, dtype='float32', mode='w+', shape=(len(word_embeds),768))
    fp[:] = word_embeds[:]
    del fp
    memmap_time = time.time() - memmap_start

    words_dict_fname = str(words_dict)[str(words_dict).index("(")+1:str(words_dict).index(")")]
    
    ## write shapes of each word_embedding job to a file to create word index later
    with open(args.outDir+'shapes.txt','a') as fout:
        fout.write(word_embeds_fname+' '+str(len(word_embeds))+'\n')
        fout.write(words_dict_fname+' '+str(len(words))+'\n')
    fout.close()

    # print stats for sanity check
    print('\n---stats---:')
    print('total time embeddings docs: %s' % (time.strftime("%H:%M:%S", time.gmtime(time_elapsed_embedding))))
    print('total time filtering content words: %s'% (time.strftime("%H:%M:%S", time.gmtime(time_elapsed_filtering))))
    print('total time creating word_sqlite_dict: %s'% (time.strftime("%H:%M:%S", time.gmtime(word_dict_time))))
    print('total elapsed copying word_embeds to memmap: %s'% (time.strftime("%H:%M:%S", time.gmtime(memmap_time))))

def create_query_matrix():

    print('creating query matrix...')
    global args

    ## xq files (query data)
    xq_fname = args.outDir+'xq.dat' ## mmep query word embeddings
    # qsents_fname = args.outDir+'qsents.pkl' ## sentences_dict
    # qwords_fname = args.outDir+'qwords.pkl' ## qwords_dict
    qsentences_dict, qwords_dict = SqliteDict(args.outDir+'qsentences.db'), SqliteDict(args.outDir+'qwords.db')

    batch_size = args.batch_size
    print('batch_size for query_matrix: ', batch_size)
    xq, q_words, q_sentences, q_trace = [], [], [], []                                           

    ## use len(query sentences as the batch_size)
    ## read in query sentences
    with open(args.query_sentences, 'r') as fin:
        for sent_id, line in enumerate(fin.read().splitlines()):
            q_sentences.append(line.strip())
            q_trace.append((args.query_sentences, sent_id))

    print('len(q_sentences) and len(q_trace): ', len(q_sentences), len(q_trace))

    spacy_docs = list(nlp.pipe(q_sentences)) ##no nead to clip len for spacy toks for the query matrix
    spacy_toks = [[tok.text for tok in doc] for doc in spacy_docs]
    spacy_pos = [[tok.pos_ for tok in doc] for doc in spacy_docs]
    spacy_deps = [[tok.dep_ for tok in doc] for doc in spacy_docs]

    ## use pytorch library DataLoader to batch sentences and helper func batchify to batch spacy annotations
    batched_q_sentences = DataLoader(q_sentences, batch_size=batch_size)
    batched_q_trace_info = DataLoader(q_trace, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_toks = DataLoader(spacy_toks, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_pos = DataLoader(spacy_pos, batch_size=batch_size, collate_fn=custom_collate)
    batched_spacy_deps = DataLoader(spacy_deps, batch_size=batch_size, collate_fn=custom_collate)

    print('DataLoader (batch_size %d): %d %d %d %d %d' %(batch_size, len(batched_q_sentences), len(batched_q_trace_info), len(batched_spacy_toks), len(batched_spacy_pos), len(batched_spacy_deps)))

    for round_id, (sentence_batch, trace_batch, spacy_toks_batch, spacy_pos_batch, spacy_deps_batch) in enumerate(zip(batched_q_sentences, batched_q_trace_info, batched_spacy_toks, batched_spacy_pos, batched_spacy_deps)):
  
        cur_words, cur_embeds = embed_sentences(round_id, sentence_batch, trace_batch, spacy_toks_batch, spacy_pos_batch, spacy_deps_batch) ## each batch is of size batch_size (see global var)


        q_words.extend(cur_words)
        xq.extend([normalize([embed])[0] for embed in cur_embeds])

    print('xq.shape: ', len(xq), len(xq[0]))

    qwords_dict_fname = str(qwords_dict)[str(qwords_dict).index("(")+1:str(qwords_dict).index(")")]

    with open(args.outDir+'shapes.txt','a') as fout:
        fout.write(xq_fname+' '+str(len(xq))+'\n')
        fout.write(qwords_dict_fname+' '+str(len(q_words))+'\n')
    fout.close()

    ## memmap qword_embeds
    fp = np.memmap(xq_fname, dtype='float32', mode='w+', shape=(len(xq),768))
    fp[:] = xq[:]
    del fp

    qsentences_dict.update([(idx,sent) for idx,sent in enumerate(q_sentences)])
    qwords_dict.update([(idx,qword) for idx,qword in enumerate(q_words)])

    qsentences_dict.commit()
    qwords_dict.commit()

    qsentences_dict.close()
    qwords_dict.close()

    print('finished processing query matrix...')

    # return xq_fname, qsents_fname, qwords_fname



                                        #################################################
                                        ###########        HELPER FUNCTIONS     #########
                                        #################################################

def get_partition_slice(it, size, section):
    '''I could change this to only return the start and end index of each subarray instead of all the indices for that partition
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))[section]

def get_slice(it, size):
    '''I could change this to only return the start and end index of each subarray instead of all the indices for that partition
    '''
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))

def custom_collate(x):
    return x    

def batchify(sentences, batch_size):

    batched_items, this_batch,  = [], []
    for cur_item in islice(sentences,None,None):
        this_batch.append(cur_item)
        if len(this_batch) == batch_size:
            batched_items.append(this_batch)
            this_batch = []
    if len(this_batch) > 0:
        batched_items.append(this_batch)

    return batched_items


def fast_read_from_sqlite_dict(sqlite_dict, start_index, end_index):

    sqlite_dict_db = sqlite3.connect(sqlite_dict)
    sqlite_dict_db_cursor = sqlite_dict_db.cursor()
    sqlite_dict_db_cursor.execute("SELECT value FROM unnamed WHERE CAST(key as INTEGER) >= ? AND CAST(key as INTEGER) <= ?;", (start_index, end_index))
    
    return [pkl.loads(x) for x in itertools.chain.from_iterable(sqlite_dict_db_cursor.fetchall())]


# import itertools
# trace_iter_1, trace_iter_2 = itertools.tee(trace_iter)
# cur_trace_data = [(value, key) for key, value in zip(trace_iter_1, fast_read_from_sqlite_dict(trace_data, trace_iter_2))]

## do sanity check in ipython on loading these dictionaries and reading in using fast read, find out how to do cur_trace
## be careful about the indexing, b/c it looks like whatever is indexed in fast read includes the end index, whereas in trace_iter = list(range(start, end)) end does not.  So you might need to do +1 or -1

                                        #################################################
                                        ###########            Main             #########
                                        #################################################                                      


def main(cur_sentences, cur_trace, cur_spacy_toks, cur_spacy_pos, cur_spacy_deps):

    global args 

    print('did you make it here?')

    ## xb files                  
    words_dict = SqliteDict(args.outDir+'words_job'+args.job_id+'.db')
    word_embeds_fname = args.outDir+'word_embeds_job'+args.job_id+'.dat'

    print('\nprocessing files for job {}...'.format(args.job_id))

    start = time.time()

    ## Generates words and respective word_embeds for each partition of the sentence index 
    ## and outputs them to outfolders to combine later for creating annoy index
    print('handling batches for job %s...' % (args.job_id))
    handle_batches(cur_sentences, cur_trace, cur_spacy_toks, cur_spacy_pos, cur_spacy_deps, words_dict, word_embeds_fname)

    handle_batches_time = time.time()-start

    print('time handling batches: %s' % (time.strftime("%H:%M:%S", time.gmtime(handle_batches_time))))

    print('finished job {}'.format(args.job_id))


if __name__ == '__main__':

    main_begin = time.time()

    print('---argparser---:')
    for arg in vars(args):
        print(arg, '\t', getattr(args, arg), '\t', type(arg))

    # run processing on GPU <gpu_id>
    cuda_idx = args.gpu_id

    with torch.cuda.device(cuda_idx):

        ## initialize bert_tokenizer and bert_model as global variable for all jobs
        if args.SBERT_flag:
            print('loading SBERT')
            ## Loads SBERT
            bert_tokenizer = None
            bert_model = SentenceTransformer('bert-base-nli-mean-tokens') ## model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
            bert_model = bert_model.cuda()
        else:
            print('loading regular BERT')
            ## Loads BERT-base uncased
            ## BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
            bert_model = bert_model.cuda()
            # bert_model = apex.amp.initialize(bert_model, opt_level="O2").to(device)

        if int(args.job_id) == 1:
            print('\nOnly processing query matrix during job {}: '.format(args.job_id))
            create_query_matrix()

        # print("l\nen(sent_data): {}, len(trace_data): {}, len(spacy_toks): {} len(spacy_pos): {} len(spacy_deps): {}".format(len(sent_data), len(trace_data), len(spacy_toks), len(spacy_pos), len(spacy_deps)))

        ## get correct partition for this job
        start_index, end_index = job_slices[int(args.job_id)-1] 
        print('\njob {} - start index: {}  end index: {} len(cur_partition): {}'.format(args.job_id, start_index, end_index, end_index-start_index))

        start = time.time()
        cur_sent_data = fast_read_from_sqlite_dict(args.outDir+args.sentences_dict, start_index, end_index)
        trace_iter = iter(list(range(start_index, end_index+1)))
        cur_trace_data = [(value, key) for key, value in zip(trace_iter, fast_read_from_sqlite_dict(args.outDir+args.trace_dict, start_index, end_index))]
        cur_spacy_toks = fast_read_from_sqlite_dict(args.outDir+args.spacy_toks_dict, start_index, end_index)
        cur_spacy_pos = fast_read_from_sqlite_dict(args.outDir+args.spacy_pos_dict, start_index, end_index)
        cur_spacy_deps = fast_read_from_sqlite_dict(args.outDir+args.spacy_deps_dict, start_index, end_index)
        retrieve_time = time.time() - start

        print('total elapsed time retrieving the current partition: %s'% (time.strftime("%H:%M:%S", time.gmtime(retrieve_time))))


        print("\nlen(cur_sent_data): {}, len(cur_trace_data): {}".format(len(cur_sent_data), len(cur_trace_data)))
        print("len(cur_spacy_toks): {} len(cur_spacy_pos): {} len(cur_spacy_deps): {}".format(len(cur_spacy_toks), len(cur_spacy_pos), len(cur_spacy_deps)))

        main(cur_sent_data, cur_trace_data, cur_spacy_toks, cur_spacy_pos, cur_spacy_deps)

        main_end = time.time() - main_begin
        print('total time inside main: %s'% (time.strftime("%H:%M:%S", time.gmtime(main_end))))

        # ## start job on partition of the sentence index
        # split_size = int(len(sent_data)/args.NUM_JOBS)
        # cur_partition = get_slice(list(range(len(sent_data))), split_size, (int(args.job_id)-1))
        # print('job {} - start index {}     end index {}'.format(args.job_id, cur_partition[0], cur_partition[-1]))
        # if len(cur_partition)>=2:
        #     i, j = cur_partition[0], cur_partition[-1]
        #     main(sent_data[i:j+1], trace_data[i:j+1], spacy_toks[i:j+1], spacy_pos[i:j+1], spacy_deps[i:j+1])
        # else:
        #     i = cur_partition[0]
        #     main(sent_data[i:], trace_data[i:], spacy_toks[i:], spacy_pos[i:], spacy_deps[i:])


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
pip install annoy
pip install faiss-gpu (see section on installing faiss for more info)
pip install sklearn
pip install -U sentence-transformers
pip install ndjson
pip install spacy
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
