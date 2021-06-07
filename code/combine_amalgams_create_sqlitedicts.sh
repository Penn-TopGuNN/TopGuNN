source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -l h=nlpgrid10
#$ -l h_vmem=100G
#$ -N combine_amalgams
python3 -u code/combine_amalgams_create_sqlitedicts.py \
-outDir 'betatest/out/' \
-dataDir 'betatest/data/' \
-NUM_JOBS 3 \
> betatest/out/combine_amalgams_create_sqlitedicts.stdout 2>&1
