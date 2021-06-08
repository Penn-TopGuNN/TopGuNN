<img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/shield-only-RGB-4k.png" align="left" width="60" hieght="60"> <img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/icon.png" align="right" /> 

# Top GuNN Similarity Search <br/>"I feel the need for speed!!!"
Aproximate Nearest Neighbors hasn't seen this speed since Maverick was in a 4 "G" inverted dive with a MIG-28... see ya in pre-flight!

![Alt Text](https://i.imgur.com/n1mR95b.gif)

<!-- ![Alt Text](https://media.giphy.com/media/GlMN2r04gXTgs/giphy.gif)

![Alt Text](https://i.pinimg.com/originals/ce/3a/c8/ce3ac8e08fa7746bd36e0ef061e9b12b.gif) -->

### Introduction

Our package, Top GuNN Similarity Search, provides python-based software to automatically query and search through approximately 5TB of vectors spanning 1000 months or articles, spanning 12-years, from 8 different news agencies, and 50GBs of the gigaword corpus. 

* I have provided data for 3 months of Gigaword, so you can betatest the scripts (Go through the README.md starting at `Order of Script Execution`).

* I have provided all of the stdouts of the results of running each file to compare against.

* I have provided a detailed README.md and additionally a file called `betatest_expected_output.pages` with computation times and file sizes of each step.

* Once you can successfully replicate my results on the 3 month betatest, you are ready to implement this on your own corpus!


### Running The Program

Running the parallelized job over AWS is divided into 4 parts:

`1) pre-processing on the CPU` 

`2) "Hot-pack" parallelized jobs on the GPU (generate BERT word embeddings and filtering for content words only).  This includes the embedding and filtering for the query matrix also!`

`3) post-processing on the CPU`

`4) Running queries on the CPU`

## Files to transfer over to AWS from the pre-processing on the CPU

	- sentences.db
	- trace.db
	- spacy_toks.db
	- spacy_pos.db
	- spacy_deps.db
	- job_slices.pkl  //list of tuples (start_index, end_index) for each partition
	- cpu_requirements.txt (in code/)
	- gpu_requirements.txt (in code/)


## Order of Script Execution
### Pre-processing on the CPU 

	TODO for each indv'l user: update your shell scripts below with you virtenv and your outDir
	TODO for each indv'l user: review global argparser vars in .py files to help you understand the command line arguments in your shell script
	NOTE: .db==sqlite_dict		.dat==npy_memmap		.pkl==pickle_file

	1.	create_amalgams_job_array.sh --> runs create_amalgams_job_array.py (which runs create_amalgams.py on each month of Gigaword in a job array simultaneously)

		sentence splits and generate SpaCy annotations for each month of Gigaword.
		each month of Gigaword will have it's own index of sentences, trace information, and spacy annotations.

		Expected output files:

			(for our case, we ran 960 jobs in parallel)
			- sentences_job#_filename.pkl (total 960)
			- trace_job#_filename.pkl (total 960)
			- spacy_toks_job#_filename.pkl (total 960)
			- spacy_pos_job#_filename.pkl (total 960)
			- spacy_deps_job#_filename.pkl (total 960)
			- create_amalgams.stdout (only 1)
			- create_amalgams_job_array_job#.stdout (1 stdout per job, we had 960 jobs, so we had 960 stdouts)
			NOTE: You do not have to print out stdouts for your jobs, we have it inserted as default in our shell scripts.  Just remove if you do not like them or need them.

	2. combine_almalgams_create_sqlitedicts.sh --> runs combine_almalgams_create_sqlitedicts.py

		Combines all the amalgamations

		Expected output files:

			(for our case, the 960 files for each are combined into 1 for each)
			- sentences.db (960 jobs into 1 sentence index)
			- trace.db (960 jobs into 1 trace information index)
			- spacy_toks.db (960 jobs into 1 spacy toks index)
			- spacy_pos.db (960 jobs into 1 spacy pos index)
			- spacy_deps.db (960 jobs into 1 spacy deps index)
			- job_slices.pkl  //list of tuples (start_index, end_index) for each partition
			- combine_amalgams_create_sqlitedicts.stdout 

	3. create_sqlite_indexes.sh --> runs create_sqlite_indexes.py

	   creates indexes to precompute the range of the partition of the amalgamated sqlite dictionary being processed for a particular job in embed_and_filter.py in the next script.

	   Expected output files:
	   		- None.  This is modifying the current amalgamated sqlite dictionaries, so that job paritions are uploaded faster and embed_and_filtering can process more rapidly for each job.
	   		- create_sqlite_indexes.stdout (if you elected to print it out in shell file)


### Parallelized job on the GPU (embedding and filtering for annoy index, and creating the query matrix)

	4.	embed_and_filter_job_launcher.py (runs embed_and_filter.py for each job for the number of jobs you want to parallelize on the GPUs)

		embeddings for both the query matrix and the database matrix are generated on AWS.
		additionally, files of the filtered words extracted in each of the jobs are outputted to files.
		NOTE: the lengths of the word embeddings and words should be equal
		NOTE: Ensure you activate your GPU environment now

		Expected output files:

		#### query matrix

		-qsentences.db   //origin query sentences
		-qwords.db       //filtered query words
		-xq.dat          //query matrix embeddings 

		#### database matrix

		-word_embeds_job#.dat	
		-words_job#.db	  //filtered words from gigaword corpus

		-shapes.txt    //memmap shapes of each of the jobs needed to load in annoy index later

		embed_and_filter_job#.stdout
		embed_and_filter_job_launcher.stdout

### Post-processing on the CPU (5a. and 5b. you can and should run simultaneously)

	5a.	create_word_index.sh --> runs create_word_index.py

		creates an amalgamated word_index and from each of the jobs that were ran in embed_and_filter_job_launcher.py in Step 4.	
		after the file is done, you should have one amalgamated word_index

		Expected output files:

		- words.db

	5b.	create_annoy_index.sh --> runs create_annoy_index.py, or
		create_faiss_index.sh --> runs create_faiss_index.py

		annoy index and faiss index are created using separate files (depending on whether you use FAISS or ANNOY).  They must have their own separate virtenvs.

		Expected output files:

		annoy_index_job#.ann    //ex. 52.32GB @ 5 months, 200 trees for subset of the Gigaword Corpus
		annoy_index_job#.ann    //ex. 52.32GB @ 3 months, 200 trees for subset of the Gigaword Corpus

# Now the Top GuNN part! Fast KNN-retrieval!

![Alt Text](https://i.imgur.com/40fvFWo.gif)

## Running queries on the CPU

	6.	synchronized_word_sim_search.sh --> runs synchronized_word_sim_search.py

		This handles a single query matrix or if you have several query matrices you want to run TopGuNN over
		If you have multiple query matrices you can indicate this with --MULTI_QUERY flag on command line.
		You must have json files of trigger words for each query matrix.
		Ex.
		Acquit.json ##[["change"], ["demoted"]]
		Sentence.json ## [["sentence"], ["sentenced"], ["sentenced"]]
		Demotion.json ## [["change"], ["demoted"]]

		Using the query matrix that was generated in embed_and_filter_job_launcher.py, each content word is queried for and synchronized searched over all the annoy or faiss indexes (whichever index you created).
		Then, all the results from all of the indexes are combined at the very end to give you the fastest similarity search possible.

		Expected output files

		- synchronized_word_sim_search_results.csv     //results in csv format of the retrieved sentences
		- synchronized_word_sim_search.stdout    //results in .txt format of the results





# Installing Dependencies...
We have provided a cpu_requirements.txt and gpu_requirements.txt in the source folder for quick install of those dependencies

## Depedencies Quick Install (Installing via cpu_requirements.txt)


	First step: Activate virtenv for cpu

	Then:
	pip3 install -r ./cpu_requirements.txt
	python3 -m spacy download en_core_web_lg
	python3 -m nltk.downloader stopwords



	First step: Activate virtenv for 

	Then:
	pip3 install -r ./gpu_requirements.txt
	python3 -m spacy download en_core_web_lg
	python3 -m nltk.downloader stopwords

	Note: if you get a YAML error, open cpu_requirements.txt or gpu_requirements.txt and comment out the line including YAML package and try running again.



## Depedencies Manual Install

### create and activate your virtual environment - you will need two for CPU processes (cpu_annoy, cpu_faiss) and two for GPU processes (gpu_annoy, gpu_faiss) if you want to use/compare results with both indexing systems.  Total of 4 virtenvs.  Otherwise, you will only need 3 virtenvs.

create: `$ python3 -m venv ~/virtenv_name/`

activate: `$ source ~/virtenv_name/bin/activate`


## ANNOY CPU virtenv:

	- pip install spacy

	`$ pip install --upgrade pip`

	`$ python3 -m spacy download en_core_web_lg`

	- pip install nltk

	`$ python3`

	`>>> import nltk`

	`>>> nltk.download('stopwords')`

	`>>> quit()')`


	- do not pip install annoy, you must git clone https://github.com/spotify/annoy.git

	`$ cd annoy/`

	`$ subl setup.py`

	`## comment lines out 1-6 in setup.py file (problematic -march flag)`

	```
	# Not all CPUs have march as a tuning parameter                                                                   
	1 cputune = ['-march=native',]                                                                                      
	2 if platform.machine() == 'ppc64le':                                                                               
	3     extra_compile_args += ['-mcpu=native',]                                                                       
	4                                                                                                                  
	5 if platform.machine() == 'x86_64':                                                                                6 
	6     extra_compile_args += cputune                                                                                 
	7                                                                                                                  
	8 if os.name != 'nt':                                                                                               
	9     extra_compile_args += ['-O3', '-ffast-math', '-fno-associative-math'] 

	```
	`$ pip install .`

	`delete repo you no longer need it`

	- pip install ndjson

	- pip install sklearn (normalize func for creating annoy index)

	- pip install faiss-cpu

	- pip install numpy

	- pip install sqlitedict

## FAISS CPU virtenv:

	- pip install spacy

	`$ pip install --upgrade pip`

	`$ python3 -m spacy download en_core_web_lg`

	- pip install nltk

	`$ python3`

	`>>> import nltk`

	`>>> nltk.download('stopwords')`

	`>>> quit()')`

	- pip install ndjson

	- pip install numpy

	- pip install faiss (on nlpgrid do not use pip install faiss-cpu)

	- pip install sklearn (due to cosine_similarity being piped from nlp_analysis.py, used for querying in word_sim_search.py)

## GPU venv:

	- pip install ndjson (still using util.py inside embed_and_filter.py)

	- pip install numpy

	- pip install tqdm

	- pip install torch (you might have to install specific torch==1.4.0 on your system)

	- pip install transformers

	- pip install sklearn (normalizing embedding in query matrix inside embed_and_filter.py)

	- pip install sentence-transformers (pip install -U sentence-transformers)

	- pip install sqlitedict

		#### Additional, (only if using faiss):

		- pip install faiss-gpu (see section on installing faiss for more info)

## More details (only use these instructions IF INSTALLING SPACY WITH THE ABOVE INSTRUCTIONS DID NOT WORK)


	- In your python environment, you must install spaCy by going to [spaCy website: linguistic features documentation](https://spacy.io/usage/linguistic-features).  
	- Although this should work with anything higher than **python 2.6**, we recommend **python 3** for running this project (I used Python 3.6 and my partner uses Python 3.5).

	> `import spaCy`

	> Change directory be in working project folder

	> use `pip install -U spacy && python -m spacy download en`

	### Common Errors
	1. *The encoding for reading in a file.*  Be sure to encode with UTF-8.  If you get an error that says something to the effect of "not recognizing ASCII characters," this is why. 
	2. *Project interpreters in your IDE (i.e. - PyCharm, Spyder).*  Ensure you cd to the working directory of your project in command line.  Once you've set the right directory in command line run "python" to ensure you are using python 2.6 or higher.  If it turns out you are not running the correct Python, you must download at this time.
	3. *Path not found, file not found.*  This means although you many have downloaded spaCy, you did not download it to the interpreter you thought you did.  You select correct python for your interpreter and then see (2) as reference.  Switch to working directly, verify which Python is being linked to the working directory, if not correct Python download accordingly. 
	4. *Cannot find module en or couldn't link model.*  You must use a seperate command to download the English model, try these two versions because it depends on an individual basis how you set up your environment and where you set the pathway to your python download (whether default or you selected custom location):
	   - `nlp = spacy.load('en')` **##this is the full english model**
	   - `nlp = spacy.load('en_core_web_sm')` **##this is the english mini-model** 
	   - This is where you will need the && `python -m spacy download en` 
	5. *pip install fail.*  This usually means you need to upgrade your pip.  Uprade your pip by using `pip install --upgrade pip` or `python.exe -m pip install --upgrade pip` in command line.  Don't forget to `cd` into the working directory of where your python is downloaded before you do this. Then once again you can try one of these commands to install spaCy after you have upgraded your pip.   
	   - `pip3 install spaCy`
	   - `pip install spaCy` 
	   - `conda install spaCy` **##sometimes pip will not work and you need to use conda**

	### More Help
	> For more help on downloading spaCy you can use [this GitHub repository](https://github.com/explosion/spaCy/issues/1721) for detailed documentation on how to deal with different issues.
