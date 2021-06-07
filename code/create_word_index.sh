source ~/annoy_cpu/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N wordmpcheck4mo
#$ -l h=nlpgrid10
#$ -l h_vmem=50G
python3 -u AWS_code/create_word_index.py \
-outDir 'AWS_code/forreal/' \
> AWS_code/forreal/create_word_index.stdout 2>&1