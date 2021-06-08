<img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/shield-only-RGB-4k.png" align="left" width="60" hieght="60"> <img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/icon.png" align="right" /> 

# TopGuNN Similarity Search <br/>"I feel the need for speed!!!"
Aproximate Nearest Neighbors hasn't seen this speed since Maverick was in a 4 "G" inverted dive with a MIG-28... see ya in pre-flight!

![Alt Text](https://i.imgur.com/n1mR95b.gif)

<!-- ![Alt Text](https://media.giphy.com/media/GlMN2r04gXTgs/giphy.gif)

![Alt Text](https://i.pinimg.com/originals/ce/3a/c8/ce3ac8e08fa7746bd36e0ef061e9b12b.gif) -->

### Introduction

Our package, TopGuNN Similarity Search, provides python-based software to automatically query and search through approximately 5TB of vectors spanning 1000 months or articles, spanning 12-years, from 8 different news agencies (Gigaword corpus is approximately 50GBs of data). 

* I have provided data for 3 months of Gigaword, so you can betatest the scripts (see `betatest/` folder, start at [Order of Script Execution](#order-of-script-execution)).

* I have provided all of the stdouts of the results from running each file to compare against.

* I have provided a detailed README.md and additionally a file called `betatest_expected_output.pages` with computation times and file sizes for each step in the pipeline so you have a frame of reference.

* Once you can successfully replicate my results on the 3 month betatest, you are ready to implement this on your own corpus!

NOTE: All computation times reported reflect running the scripts on our Penn NLP clusters, which use spinning disk, so times might vary slightly depending on your resources.


## Table of Contents 

- [Pre-processing on the CPU](#pre-processing-on-the-CPU)
- [Parallelized jobs on the GPU](#parallelized-jobs-on-the-GPU)
- [Post-processing on the CPU](#post-processing-on-the-CPU)
- [Running queries on the CPU](#running-queries-on-the-CPU)


# Order of Script Execution 
## Pre-processing on the CPU 

	TODO: Update your shell scripts below with your virtenv and your outDir/dataDir
	TODO: Activate your CPU environment now
	NOTE: .db==sqlite_dict		.dat==npy_memmap		.pkl==pickle_file

	1.	create_amalgams_job_array.sh --> runs create_amalgams_job_array.py 

		Sentence splits and generates SpaCy annotations for each month of Gigaword.
		Each month of Gigaword will have it's own index of sentences, trace information, and spacy annotations.

		Expected output files:

			(We ran 960 jobs in parallel)
			- sentences_job#_filename.pkl (total 960)
			- trace_job#_filename.pkl (total 960)
			- spacy_toks_job#_filename.pkl (total 960)
			- spacy_pos_job#_filename.pkl (total 960)
			- spacy_deps_job#_filename.pkl (total 960)
			- create_amalgams.stdout (only 1)
			- create_amalgams_job_array_job#.stdout (total 960)


	2. combine_almalgams_create_sqlitedicts.sh --> runs combine_almalgams_create_sqlitedicts.py

		Combines all the amalgamations

		Expected output files:

			(The 960 files for each are combined into 1 for each)
			- sentences.db (960 jobs into 1 sentence index)
			- trace.db (960 jobs into 1 trace information index)
			- spacy_toks.db (960 jobs into 1 spacy toks index)
			- spacy_pos.db (960 jobs into 1 spacy pos index)
			- spacy_deps.db (960 jobs into 1 spacy deps index)
			- job_slices.pkl  //list of tuples (start_index, end_index) for each partition
			- combine_amalgams_create_sqlitedicts.stdout 

	3. create_sqlite_indexes.sh --> runs create_sqlite_indexes.py

	   creates indexes to precompute the range of the partition of the amalgamated sqlite dictionary being processed for a particular job in embed_and_filter_job_array.py.

	   Expected output files:
	   		- None.  This is modifying the current amalgamated sqlite dictionaries, so that job paritions are uploaded faster and embed_and_filtering_job_array.py can process more rapidly for each job.
	   		- create_sqlite_indexes.stdout 

	   	Now you are ready to transfer all the files from Step 2) over to AWS for embedding and filtering!


## Parallelized jobs on the GPU 

	4.	embed_and_filter_job_launcher.py 

		TODO: Activate your GPU environment now
		Embeddings for both the query matrix and the database matrix are generated here.
		Files of the filtered words extracted in each of the jobs are outputted to files.	

		Expected output files:

		#### query matrix

		-qsentences.db   //origin query sentences
		-qwords.db       //filtered query words
		-xq.dat          //query matrix embeddings 

		#### database matrix

		-word_embeds_job#.dat	
		-words_job#.db	  //filtered words from gigaword corpus

		#### other

		-shapes.txt    //memmap shapes of each of the jobs needed to load in annoy index later

		embed_and_filter_job#.stdout
		embed_and_filter_job_launcher.stdout

## Post-processing on the CPU 

	TODO: Activate your CPU environment now
	(5a. and 5b. run scripts simultaneously)

	5a.	create_word_index.sh --> runs create_word_index.py

		Creates an amalgamated word_index and from each of the jobs that were ran in embed_and_filter_job_launcher.py in Step 4.	
		After the file is done, you should have one amalgamated word_index

		Expected output files:

		- words.db

	5b.	create_annoy_index.sh --> runs create_annoy_index.py

		Expected output files:
		annoy_index_job#.ann    //ex. 7.1-7.3GB @ for each job, assuming around 200K sentences each.

# Now the TopGuNN part! Fast KNN-retrieval!

![Alt Text](https://i.imgur.com/40fvFWo.gif)

## Running queries on the CPU

	6.	synchronized_word_sim_search.sh --> runs synchronized_word_sim_search.py

		This supports a single query matrix or several query matrices
		If you have multiple query matrices you can indicate this with --MULTI_QUERY flag on command line.
		You must have json files of trigger words for each query matrix.
			Ex.
			Acquit.json ##[["change"], ["demoted"]]
			Sentence.json ## [["sentence"], ["sentenced"], ["sentenced"]]
			Demotion.json ## [["change"], ["demoted"]]

		Expected output files:

		- synchronized_word_sim_search_results.csv     //results in csv format of the retrieved sentences
		- synchronized_word_sim_search.stdout    //results in .txt format of the results

