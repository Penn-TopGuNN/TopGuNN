source ~/annoy_cpu/bin/activate
# TODO: annoy index, stdout, and csv filenames should match for every run_id (for organization)
# TODO: switch out -annoy_index for -faiss_index && --ANNOY for --FAISS flag command line args depending on which sim search you're doing
# TODO: switch out faiss and annoy imports in py script
#!/bin/zsh
#$ -cwd
#$ -N simsrch14out
#$ -l h=nlpgrid10
#$ -l h_vmem=50G
python3 -u AWS_code/word_sim_search.py -outDir 'AWS_code/14out_5mo_pkl_slices/' -top_n 50 -num_trees 160 --ANNOY -sentences_dict 'sentences.db' -trace_dict 'trace.db' -annoy_index 'annoy_index.ann' -csv_file 'eventprimitives_pruned.csv' -words_dict 'words.db' -xq 'xq.dat' -qsents 'qsentences.db' -qwords 'qwords.db' > 'AWS_code/14out_5mo_pkl_slices/word_sim_search.stdout' 2>&1 