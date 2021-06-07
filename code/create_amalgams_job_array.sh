source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -l h=(nlpgrid10|nlpgrid11|nlpgrid12|nlpgrid15|nlpgrid16|nlpgrid19)
#$ -l mem=4G
#$ -N TopGuNN
#$ -t 1-3
python3 -u code/create_amalgams_job_array.py \
-job_id $SGE_TASK_ID \
-outDir 'betatest/out/' \
-dataDir 'betatest/data/' \
> betatest/out/create_amalgams_job_array_job$SGE_TASK_ID.stdout 2>&1