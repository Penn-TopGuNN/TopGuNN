import pickle as pkl
import json
from functools import reduce
from itertools import starmap, compress, chain, repeat, islice
import gzip
import ndjson
import numpy as np
import time

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')
nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

def pretty_time(xtime):
    return time.strftime("%H:%M:%S", time.gmtime(xtime))

def pretty_print_json(json_filename):
    data = json_load(json_filename)
    print(json.dumps(data, indent=4, sort_keys=False))

def json_dump(data, output_json_file):
    with open(output_json_file, 'w') as fout:
        json.dump(data, fout)

def json_load(filename):
    with open(filename, 'r') as fin:
        data = json.load(fin)
    fin.close()
    return data

def ndjson_dump(data,fname):
    # dump to file-like objects
    with open(fname, 'w') as f:
        ndjson.dump(data, f)

def ndjson_load(fname):
    # load from file-like objects
    with open(fname) as f:
        data = ndjson.load(f)
    return data

def np_save(data, fname):
    with open(fname, 'wb') as f:
        np.save(f, data)

def np_load(fname):
    with open(fname, 'rb') as fin:
        data = np.load(fin)
    return data

def pickle_dump(data, output_pickle_file):
    with open(output_pickle_file, 'wb') as fout:
        pkl.dump(data, fout)

def pickle_load(filename):
    with open(filename, 'rb') as fin:
        data = pkl.load(fin)
    fin.close()
    return data

def gopen(fname):
    with gzip.open(fname, "rt") as fin: 
        data = fin.read().splitlines()
        ## (list) 
        return data


def serialize_index(index):
    """ convert an index to a numpy uint8 array  """
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return faiss.vector_to_array(writer.data)

def deserialize_index(data):
    reader = faiss.VectorIOReader()
    faiss.copy_array_to_vector(data, reader.data)
    return faiss.read_index(reader)

def write_outfile(data,filename):
    with open(filename, 'w') as fout:
        for item in data:
            fout.write(item+'\n')

def flatten(indices):
    flat = []
    list(map(lambda subword: flat.extend(subword[1:-1]),indices))
    return [101]+flat+[102]

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size) 

class MaskableList(list):
    def __getitem__(self, index):
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            return MaskableList(compress(self, index))
