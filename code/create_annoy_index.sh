## TODO: update run_id, annoy indexes filenames for each of the jobs, and stdout - don't overwrite
source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N TopGuNN_create_annoy_index
#$ -l h_vmem=100G
#$ -l h=nlpgrid10
python3 -u code/create_annoy_index.py \
-outDir 'betatest/out/' \
-annoy_run_id 'annoy_index_' \
-nworkers 3 \
-batch_size 400000 \
> 'betatest/out/create_annoy_index.stdout' 2>&1
