source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -l h=nlpgrid10
#$ -l h_vmem=400G
#$ -N bigawsjob
python3 -u AWS_code/create_amalgams.py \
-outDir 'AWS_code/bigjob/iterative_create_amalgams_rev2/' \
-dataDir '/nlp/data/corpora/LDC/LDC2011T07/data/' \
-NUM_JOBS 960 \
> AWS_code/bigjob/iterative_create_amalgams_rev2/create_amalgams.stdout 2>&1