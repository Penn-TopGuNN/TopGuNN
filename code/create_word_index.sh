source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N TopGuNN_create_word_index
#$ -l h=nlpgrid10
#$ -l h_vmem=50G
python3 -u code/create_word_index.py \
-outDir 'betatest/out/' \
> betatest/out/create_word_index.stdout 2>&1