source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -l h=nlpgrid10
#$ -l h_vmem=50G
#$ -N TopGuNN
python3 -u code/create_amalgams.py \
-outDir 'betatest/out/' \
-dataDir 'betatest/data/' \
-NUM_JOBS 3 \
> betatest/out/create_amalgams.stdout 2>&1