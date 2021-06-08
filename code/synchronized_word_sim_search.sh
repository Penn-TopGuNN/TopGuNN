source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N synchronized_word_sim_search
#$ -l h=nlpgrid10
#$ -l h_vmem=150G
python3 -u code/synchronized_word_sim_search.py \
-num_workers 5 \
-queryDir 'betatest/out/' \
-outDir 'betatest/out/' \
-annoy_run_id 'annoy_index' \
-top_n 300 \
-top_n_increment 150 \
-top_n_threshold 1000 \
-num_uniq 8 \
-acc 1 \
--SINGLE_QUERY \
> 'betatest/out/synchronized_word_sim_search.stdout' 2>&1 
