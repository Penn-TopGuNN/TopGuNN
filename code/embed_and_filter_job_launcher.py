import multiprocessing
from multiprocessing import Pool
# from multiprocessing import Pool, Lock
import os
import random
import time

'''usage:
python3 -u code/embed_and_filter_job_launcher.py \
'''

## big AWS Job
# NUM_GPUS = 16
# NUM_JOBS = 960 
# GPU_IDS = list(range(17))

## for 3 month betatest
NUM_GPUS = 3
NUM_JOBS = 3 # 3 month test run
GPU_IDS = list(range(1,4)) # [1,2,3]


def job(job_args):

    gpu_locks, i = job_args

    gpu_id = None
    gpu_lock = None
    for gpu_lock_idx, lock in enumerate(gpu_locks):
        if lock.acquire(False):
            gpu_id = GPU_IDS[gpu_lock_idx]
            gpu_lock = lock
            break
        else:
            # Continue loop, try to find another GPU that isn't locked
            pass

    # In case, we make it here, without finding a free GPU
    # safely, wait until one is free
    if gpu_id is None or gpu_lock is None:
        random_gpu_lock_idx = random.choice(range(NUM_GPUS))
        lock = gpu_locks[random_gpu_lock_idx]
        lock.acquire() # wait until GPU 0 is free....
        gpu_id = GPU_IDS[random_gpu_lock_idx]
        gpu_lock = lock

    print("starting job %d..." % (i,))
    
    os.system('''python3 -u code/embed_and_filter.py -job_id {} -outDir 'betatest/out/' -dataDir 'betatest/data/' -gpu_id {}  -batch_size 175 -clip_len 225 -job_slices "job_slices.pkl" -query_sentences 'betatest/data/query_sentences.txt' -sentences_dict 'sentences.db' -trace_dict 'trace.db' -spacy_toks_dict 'spacy_toks.db' -spacy_pos_dict 'spacy_pos.db' -spacy_deps_dict 'spacy_deps.db' --BERT --MEAN > 'betatest/out/embed_and_filter_job{}.stdout' 2>&1'''.format(i, gpu_id, i))
    print('finished bash script..')

    # Free up the GPU for other future proccesses
    gpu_lock.release()



if __name__ == '__main__':

    start = time.time()
    m = multiprocessing.Manager()

    gpu_locks = []
    for i in range(NUM_GPUS):
        gpu_locks.append(m.Lock())

    with Pool(NUM_GPUS) as p:
        job_ids = range(1, NUM_JOBS + 1)
        gpu_locks_per_job = [gpu_locks] * NUM_JOBS

        p.map(job, list(zip(gpu_locks_per_job, job_ids)))

    main_time = time.time() - start
    print('finished job launcher script...')
    print(f'Time elapsed in embed_and_filter_job_launcher.py:\t{time.strftime("%H:%M:%S", time.gmtime(main_time))}')



