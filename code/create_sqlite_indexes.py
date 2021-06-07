import argparse
import sqlite3
import time

parser = argparse.ArgumentParser(description='creating indexes of sqlite dicts...')
parser.add_argument('-outDir', required=True, help='Directory where all outfiles will be written to. Example: out/')
parser.add_argument('-sentences_dict', required=True, help="sqlite db filename. Example: 'sentences_dict.db'")
parser.add_argument('-trace_dict', required=True, help="sqlite db filename. Example: 'trace_dict.db'")
parser.add_argument('-spacy_toks_dict', required=True, help="sqlite db filename. Example: 'spacy_toks_dict.db'")
parser.add_argument('-spacy_pos_dict', required=True, help="sqlite db filename. Example: 'spacy_pos_dict.db'")
parser.add_argument('-spacy_deps_dict', required=True, help="sqlite db filename. Example: 'spacy_deps_dict.db'")
args = parser.parse_args()

def create_sqlite_index(fname):
  print('Creating SQLite Index on ... %s' % (fname,))
  sqlite_begin = time.time()
  sqlite_dict_db = sqlite3.connect(fname)
  sqlite_dict_db_cursor = sqlite_dict_db.cursor()
  sqlite_dict_db_cursor.execute("CREATE INDEX key_int ON unnamed(CAST(key as INTEGER));")
  sqlite_dict_db.commit()
  sqlite_dict_db.close()
  sqlite_end = time.time() - sqlite_begin
  print('total time inside create_sqlite_index: %s'% (time.strftime("%H:%M:%S", time.gmtime(sqlite_end))))

if __name__ == '__main__':
  main_begin = time.time()
  create_sqlite_index(args.outDir+args.sentences_dict)
  create_sqlite_index(args.outDir+args.trace_dict)
  create_sqlite_index(args.outDir+args.spacy_toks_dict)
  create_sqlite_index(args.outDir+args.spacy_pos_dict)
  create_sqlite_index(args.outDir+args.spacy_deps_dict)
  main_end = time.time() - main_begin
  print("Done creating all SQLite indexes.")
  print('total time inside main: %s'% (time.strftime("%H:%M:%S", time.gmtime(main_end))))


