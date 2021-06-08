source ~/topgun/bin/activate
#!/bin/zsh
#$ -cwd

N=3
for i in {1..3}; do
    (
        echo "starting job $i..."
        python3 -u AWS_code/embed_and_filter.py \
        -job_id $i \
        -outDir 'code/out/' \
        -dataDir 'betatest/data/' \
        -NUM_JOBS 3 \
        -NUM_GPUS 3 \
        -PROC_PER_GPU 1 \
        -gpu_ids 1 2 3 \
        -batch_size 175 \
        -clip_len 225 \
        -job_slices "job_slices.pkl" \
        -query_sentences 'betatest/data/query_sentences.txt' \
        -sentences_dict 'sentences.db' \
        -trace_dict 'trace.db' \
        -spacy_toks_dict 'spacy_toks.db' \
        -spacy_pos_dict 'spacy_pos.db' \
        -spacy_deps_dict 'spacy_deps.db' \
        --BERT \
        --MEAN \
        > 'betatest/out/embed_and_filter_job_handler_job'$i'.stdout' 2>&1
    ) &

    # allow only to execute $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
        # wait only for first job
        wait -n
    fi

done

# wait for pending jobs
wait

echo "bash script completed processing..."
