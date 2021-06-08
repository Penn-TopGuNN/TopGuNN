import argparse
from os import listdir, walk
from os.path import isfile, join
import gzip
import nltk
import util
from nltk.tokenize import sent_tokenize
import spacy
import time
nlp = spacy.load("en_core_web_lg", disable=["ner"])


'''usage: (goes in shell script or copy&paste into command line)
python3 -u code/create_amalgams_job_array.py \
-job_id $SGE_TASK_ID \
-outDir 'betatest/out/' \
-dataDir 'betatest/data/' \
> code/out/create_amalgams_job_array_job$SGE_TASK_ID.stdout 2>&1
'''
## global argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-job_id', type=str, required=True, help='job number being processes')
parser.add_argument('-outDir', required=True, help='example: out/')
parser.add_argument('-dataDir', required=True, help="example: 'data/'' or '/nlp/data/corpora/LDC/LDC2011T07/data/'")
parser.add_argument('-clip_len', type=int, default=225, required=False, help='example: 225')
args = parser.parse_args()

## Reads in a sample gigaword file (non-annotated)
def main(gigaword_fname):
    '''Reads in a sample gigaword file (non-annotated, 1 month of NYT)
    Args:
    Returns:
    Note:
    '''
    global args, sentences, trace
    counter = 0
    sentences, trace = [], []
    total_ndocs, total_nsents = 0, 0
    cur_text, doc_id, begin_tok = '', '', ''

    start = time.time()
    print('entering main...')

    with gzip.open(gigaword_fname, 'rt') as f: 
        for i, line in enumerate(f):
            ## Saves document id
            if line.startswith('<DOC id='):
                doc_id = line.strip()[9:30]
                total_ndocs += 1
            elif line.startswith('<P>'):
                begin_tok = line.strip()
            elif line.startswith('</P>'):
                continue
            elif line.startswith('</TEXT>'):
                if len(cur_text) != 0:
                    cur_sents = sent_tokenize(cur_text)
                    total_nsents += (len(cur_sents))
                    sentences.extend([sent.strip() for sent in cur_sents])
                    trace.extend([(doc_id,sent_id) for sent_id in range(len(cur_sents))])                 
                    cur_text, begin_tok = '', ''
                    # test for add 10 sents per file for debugging
                    # if total_nsents > 10:
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
    print('starting spacy annotations..')
    spacy_docs = list(nlp.pipe(sentences)) 
    spacy_toks = [[tok.text for tok in doc][:args.clip_len] for doc in spacy_docs]
    spacy_pos = [[tok.pos_ for tok in doc][:args.clip_len] for doc in spacy_docs]
    spacy_deps = [[tok.dep_ for tok in doc][:args.clip_len] for doc in spacy_docs]
    print('end spacy annotations..')
    print('\nRead File Stats - total_ndocs: {}  total_nsents: {} total_ntrace: {} total_nspacy_docs: {}'.format(total_ndocs, total_nsents, len(trace), len(spacy_docs)))
    spacy_time = time.time() - start

    print('Time elapsed reading file for job {}:\t{}'.format(args.job_id, time.strftime("%H:%M:%S", time.gmtime(read_time))))
    print('Time elapsed spacy annotations for job {}:\t{}'.format(args.job_id, time.strftime("%H:%M:%S", time.gmtime(spacy_time))))

    util.pickle_dump(sentences, args.outDir+'sentences_job'+args.job_id+'_'+gigaword_fname[-17:-3]+'.pkl')
    util.pickle_dump(trace, args.outDir+'trace_job'+args.job_id+'_'+gigaword_fname[-17:-3]+'.pkl')
    util.pickle_dump(spacy_toks, args.outDir+'spacy_toks_job'+args.job_id+'_'+gigaword_fname[-17:-3]+'.pkl')
    util.pickle_dump(spacy_pos, args.outDir+'spacy_pos_job'+args.job_id+'_'+gigaword_fname[-17:-3]+'.pkl')
    util.pickle_dump(spacy_deps, args.outDir+'spacy_deps_job'+args.job_id+'_'+gigaword_fname[-17:-3]+'.pkl')

def job_handler():

    global args
    # fnames = sorted(["data/"+fname for fname in os.listdir("data/") if fname.startswith("nyt_eng")])
    # print('nfiles: ', len(fnames))
    # print('files: ', fnames)

    ## LDC CORPORA
    ## /nlp/data/corpora/LDC/LDC2011T07/data/
    ## sorted([join(rootdir+'/',fname) for (rootdir,subdir,fnames) in list(walk(dataDir))[1:] for fname in fnames])
    fnames = sorted([join(args.dataDir,subdir,fname) for subdir in listdir(args.dataDir) for fname in listdir(args.dataDir+subdir)]) 
    print('\nnfiles: ', len(fnames))
    print('\nfnames: ')
    print(fnames)

    gigaword_fname = fnames[int(args.job_id)-1]
    print('\ncurrent file being processed: {}'.format(gigaword_fname[-17:]))
    main(gigaword_fname)

if __name__ == '__main__':

    main_start = time.time()

    print('---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    job_handler()

    main_time = time.time() - main_start
    print('\nfinished processing jobs...')
    print('Time elapsed in main for job {}: {}'.format(args.job_id, time.strftime("%H:%M:%S", time.gmtime(main_time))))



