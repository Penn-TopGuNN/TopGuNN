source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N triggerwords_simsearch
#$ -l h=nlpgrid10
#$ -l h_vmem=150G
python3 -u AWS_code/synchronized_word_sim_search.py \
-num_workers 5 \
-queryDir 'betatest_queryDir/' \
-outDir 'AWS_code/14out_5mo_pkl_slices/' \
-annoy_run_id 'annoy_index_pool' \
-top_n 1000 \
-top_n_increment 500 \
-top_n_threshold 3000 \
-num_uniq 10 \
-acc 1 \
> 'betatest_out/mp_word_sim_search_new_trigger_word_logic.stdout' 2>&1 
