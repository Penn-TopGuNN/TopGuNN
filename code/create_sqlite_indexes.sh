source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -l h_vmem=100G
#$ -l h=nlpgrid10
#$ -N sqlite_indexes
python3 -u AWS_code/create_sqlite_indexes.py \
-outDir 'AWS_code/forreal/' \
-sentences_dict 'sentences.db' \
-trace_dict 'trace.db' \
-spacy_toks_dict 'spacy_toks.db' \
-spacy_pos_dict 'spacy_pos.db' \
-spacy_deps_dict 'spacy_deps.db' \
> 'AWS_code/forreal/create_sqlite_indexes.stdout' 2>&1