import gzip, json, sys
import numpy as np
from collections import defaultdict, Counter
import annoy
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
import time
import argparse
import util
import os
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import numpy as np
## TODO: Line 138 determine which normalization method to use

'''usage for annoy (goes into the shell script):
python3 -u create_annoy_index.py \
-annoy_index 'annoy_index_run_id.ann'
-outdir 'AWS_code/1out/' \
-num_trees 640 \

num_trees = max(50, int((number_of_vectors / 3000000.0) * 50.0))
search_k = top_n * num_trees

## Expected nembeddings: 9634502

'''

'''global argparser'''
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', type=str, required=True, help='Directory where all outfiles will be written to.')
parser.add_argument('-annoy_run_id', type=str, required=True, help='How will the annoy indexes be named (the prefix)?')
parser.add_argument('-dim', default=768, type=int, required=False, help='Number of dimesions of the embeddings.')
parser.add_argument('-nworkers', default=4, type=int, required=True, help='Number of workers for pooling the annoy indexes.')
parser.add_argument('-batch_size', default=200000, type=int, required=True, help='Number of embeddings to batch.')
args = parser.parse_args()

'''global variables'''
data = open(args.outDir+"shapes.txt","r")
shapes = data.read().splitlines()
shapes_dict = {}
for item in shapes: 
    fname,length = item.split(" ") 
    shapes_dict[fname]=length 
data.close()


def create_annoy_index(fname):

    fname_abv = fname.split('/')[-1]
    print('Processing file {}...'.format(fname_abv))

    global args, shapes_dict

    word_embeds_data = np.memmap(fname, dtype='float32', mode='r', shape=(int(shapes_dict[fname]),768))

    ## add normalized embeddings to an annoy index
    start = time.time()
    print('\nfile {} \tlen(word_embeds_data) {} '.format(fname_abv, len(word_embeds_data)))
    annoy_index = AnnoyIndex(args.dim, 'euclidean') 
    annoy_index.on_disk_build(args.outDir+args.annoy_run_id+fname_abv.split('.')[0].split('_')[-1]+'.ann')

    for i in range(0, len(word_embeds_data), 200000):
        start_idx = i
        end_idx = i + 200000
        embeds_batch = word_embeds_data[start_idx:end_idx]
        lengths = np.linalg.norm(embeds_batch, axis=1).reshape((len(embeds_batch),1))
        normalized_embeds_batch = embeds_batch / lengths 
        for j, embedding in enumerate(normalized_embeds_batch, start_idx):
            annoy_index.add_item(j, embedding)
            if j % 10000 == 0:
                print('file {} \tadded {} embeddings to annoy ...'.format(fname_abv, j))
    add_time = time.time() - start

    num_trees = max(50, int(len(word_embeds_data)/3000000) * 50)
    print('file {} \tbuilding annoy_index, ntrees {}'.format(fname_abv, num_trees))
    start = time.time()
    annoy_index.build(num_trees)
    build_time = time.time() - start
    print('file {} \tfinished building annoy index'.format(fname_abv))

    print('file {} \tword_embeds_data.shape {}'.format(fname_abv,len(word_embeds_data)))
    print('file {} \tannoy_index.ntrees {}'.format(fname_abv,annoy_index.get_n_trees()))
    print('file {} \tannoy_index.ntotal {}'.format(fname_abv,annoy_index.get_n_items()))

    print('file {} \tTime elapsed adding individual embedding to annoy index:\t{}'.format(fname_abv, (time.strftime("%H:%M:%S", time.gmtime(add_time)))))
    print('file {} \tTime elapsed building annoy index:\t{}'.format(fname_abv,(time.strftime("%H:%M:%S", time.gmtime(build_time)))))
    print('Finished processing file {}...'.format(fname_abv))


if __name__ == '__main__':

    start = time.time()

    print('---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg))


    word_embeds_fnames = sorted([join(args.outDir,f) for f in listdir(args.outDir) if f.startswith('word_embeds_job')])
    print('len(word_embeds_fnames): ', len(word_embeds_fnames))
    print('check word_embeds_fnames: ')
    for fname in word_embeds_fnames:
        print(fname)

    
    with Pool(args.nworkers) as p:
        p.map(create_annoy_index, word_embeds_fnames) 

    main_time = time.time() - start
    print('finished annoy index job launcher in main...')
    print(f'Time elapsed building annoy index in main:\t{time.strftime("%H:%M:%S", time.gmtime(build_time))}')


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
