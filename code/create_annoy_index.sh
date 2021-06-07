## TODO: update run_id, annoy indexes filenames for each of the jobs, and stdout - don't overwrite
source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N annoypool400
#$ -l h_vmem=100G
#$ -l h=nlpgrid10
python3 -u AWS_code/create_annoy_index.py \
-outDir 'AWS_code/14out_5mo_pkl_slices/' \
-annoy_run_id 'annoy_index_' \
-nworkers 3 \
-batch_size 400000 \
> 'AWS_code/pooling_create_annoy_indexes_400K_Pool3_rev2.stdout' 2>&1
