from collections import OrderedDict
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from threading import Lock
from time import sleep
import time
import math

thread_pools = []

def init(num_workers, _num_synchronous_executions, _log = False):
  global pool, thread_pools, sleep_delay, cur_iter, jobs_waiting
  global job_submitted, num_done, job_submissions, job_sync_lock
  global job_submission_lock, job_results, num_synchronous_executions
  global log, num_done_last_updated, last_execution_time, done_lock

  pool = Pool(num_workers)

  # Global variables accessible in multiple threads (but not multiple processes)
  log = _log
  num_synchronous_executions = _num_synchronous_executions
  sleep_delay = .1
  cur_iter = 0
  jobs_waiting = 0
  job_submitted = 0
  num_done = {}
  last_execution_time = time.time()
  num_done_last_updated = {}
  job_submissions = OrderedDict()
  job_sync_lock = Lock()
  job_submission_lock = Lock()
  job_results = OrderedDict()

def synchronize_over(synchronized_func, synchronized_over):
  global thread_pools
  thread_pool = ThreadPool(len(synchronized_over))
  thread_pool.map(synchronized_func, enumerate(synchronized_over))
  thread_pools.append(thread_pool)


def execute_jobs_if_ready(job_func, synchronized_idx):
  global job_results, job_submitted, job_submissions, last_execution_time

  # Wait for previous executions to finish before continuing
  while not all([time > last_execution_time for time in num_done_last_updated.values()]):
    sleep(sleep_delay)

  # Only continue, if all jobs are submitted
  if job_submitted == (num_synchronous_executions - sum(num_done.values())):
    first_time = time.time()
    job_results = OrderedDict(zip(list(job_submissions.keys())[0:1], pool.map(job_func, list(job_submissions.values())[0:1]))) # Just do the first submission on its own to warm up the cache
    first_time = time.time() - first_time
    subsequent_time = time.time()
    job_results.update(zip(list(job_submissions.keys())[1:], pool.map(job_func, list(job_submissions.values())[1:]))) # Then do the rest with the warmed up cache
    subsequent_time = (time.time() - subsequent_time) / (len(job_submissions) - 1) if len(job_submissions) > 1 else 0
    if log:
      print('Time elapsed for first query:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(first_time))))
      print('Time elapsed for subsequent queries:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(subsequent_time))))

    # Reset state
    old_job_submitted = job_submitted
    job_submitted = 0
    job_submissions = OrderedDict()

  else:
    # Not ready yet
    pass

# Args is a list of arg tuples, the first item in each tuple must be the index of the job
def map(synchronized_idx, job_func, args):
  global cur_iter, jobs_waiting, job_submitted, num_done, last_execution_time, num_done_last_updated

  results = []
  for arg_idx, arg in enumerate(args):
    i = arg[0]
    while i != cur_iter:
      sleep(sleep_delay)
    job_submissions[synchronized_idx] = arg
    master = job_sync_lock.acquire(False)
    jobs_waiting += 1
    job_submitted += 1
    job_submission_lock.acquire()
    execute_jobs_if_ready(job_func, synchronized_idx)
    job_submission_lock.release()
    if master:
      while True:
        if job_submitted == 0 and jobs_waiting == 1:
          jobs_waiting -= 1
          job_sync_lock.release()
          cur_iter = (cur_iter + 1) % len(args)
          break
        sleep(sleep_delay)
    else:
      while True:
        if job_submitted == 0:
          jobs_waiting -= 1
          break
        sleep(sleep_delay)

    result = job_results[synchronized_idx]
    results.append(result)

  # Before returning the results, we set
  # the last_updated time to be less than
  # the new last_execution_time so that
  # execute_jobs_if_ready() can wait
  # until all last_updated's > last_execution_time
  num_done_last_updated[synchronized_idx] = time.time()
  last_execution_time = time.time() 
  sleep(5) # Slightly, hacky solution, if things start freezing, just increase value
  return results

def mark_as_done(synchronized_idx):
  global num_done
  num_done[synchronized_idx] = True
  num_done_last_updated[synchronized_idx] = float('inf')

def mark_as_not_done(synchronized_idx):
  global num_done
  num_done[synchronized_idx] = False
  num_done_last_updated[synchronized_idx] = time.time()

def close():
  for thread_pool in thread_pools:
    thread_pool.close()
  pool.close()
