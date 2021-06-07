<img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/shield-only-RGB-4k.png" align="left" width="60" hieght="60"> <img src="https://github.com/bikegirl/CIT591-NLP-Nitro/blob/master/media/icon.png" align="right" /> 

# Top GuNN Similarity Search <br/>"I feel the need for speed!!!"
Aproximate Nearest Neighbors hasn't seen this speed since Maverick was in a 4 "G" inverted dive with a MIG-28... see ya in pre-flight!

![Alt Text](https://i.imgur.com/n1mR95b.gif)

<!-- ![Alt Text](https://media.giphy.com/media/GlMN2r04gXTgs/giphy.gif)

![Alt Text](https://i.pinimg.com/originals/ce/3a/c8/ce3ac8e08fa7746bd36e0ef061e9b12b.gif) -->

### Introduction

Our package, Top GuNN Similarity Search, provides python-based software to automatically query and search through vectors spanning 1000 months over 12-years, 8 different news agencies, and 50GBs of the gigaword corpus (between 10G - 1T vectors). 

### Running The Program

Running the parallelized job over AWS is divided into 4 parts:

`1) pre-processing on the CPU` 

`2) "Hot-pack" parallelized jobs on the GPU (generate BERT word embeddings and filtering for content words only).  This includes the embedding and filtering for the query matrix also!`

`3) post-processing on the CPU`

`4) Running queries on the CPU`

## Files to transfer over to AWS from nlpgrid

	- sentences.db
	- trace.db
	- spacy_toks.db
	- spacy_pos.db
	- spacy_deps.db
	- job_slices.pkl  //list of tuples (start_index, end_index) for each partition
	- cpu_requirements.txt (in AWS_code/)
	- gpu_requirements.txt (in AWS_code/)


## Order of script execution
### Pre-processing on the CPU (run on NLPGrid)

	TODO for each indv'l user: update your shell scripts below with you virtenv and your outDir
	TODO for each indv'l user: review global argparser vars in .py files to help you understand the command line arguments in your shell script
	NOTE: .db==sqlite_dict		.dat==npy_memmap		.pkl==pickle_file

	1.	create_amalgams.py / create_amalgams.sh

		creates an amalgamation of each file for sentences, trace information, and spacy annotations.

		Expected output files:

			- sentences.db
			- trace.db
			- spacy_toks.db
			- spacy_pos.db
			- spacy_deps.db
			- job_slices.pkl  //list of tuples (start_index, end_index) for each partition

	2. create_sqlite_indexes.py / create_sqlite_index.sh

	   creates indexes to precompute the range of the partition of the amalgamated sqlite dictionary being processed for a particular job in embed_and_filter.py in the next script.

	   Expected output files:
	   		- None.  This is modifying the current amalgamated sqlite dictionaries, so that job paritions are uploaded faster and embed_and_filtering can process more rapidly for each job.


### Distributed job on the GPU (includes: embedding and filtering for annoy index, and creating the query matrix)

	3.	embed_and_filter.py / embed_and_filter.sh

		embeddings for both the query matrix and the database matrix are generated on AWS.
		additionally, files of the filtered words extracted in each of the jobs are outputted to files.
		NOTE: the lengths of the word embeddings and words should be equal

		Expected output files:

		#### query matrix

		-qsentences.db   //origin query sentences
		-qwords.db       //filtered query words
		-xq.dat          //query matrix embeddings 

		#### database matrix

		-word_embeds_job#.dat	
		-words_job#.db	  //filtered words from gigaword corpus

		-shapes.txt    //memmap shapes of each of the jobs needed to load in annoy index later

### Post-processing on the CPU (4a. and 4b. you can and should run simultaneously)

	4a.	create_word_index.py / create_word_index.sh

		creates an amalgamated word_index and from each of the jobs that were ran in embed_and_filter.py
		after the file is done, you should have one amalgamated word_index

		Expected output files:

		- words.db

	4b.	create_annoy_index.py, create_annoy_index.sh  / create_faiss_index.py, create_faiss_index.sh

		annoy index and faiss index are created using separate files (depending on whether you use FAISS or ANNOY).  They must have their own separate virtenvs.

		Expected output files:

		annoy_index.ann    //52.32GB @ 5 months, 200 trees

# Now the Top GuNN part! Fast KNN-retrieval!

![Alt Text](https://i.imgur.com/40fvFWo.gif)

## Running queries on the CPU

	4.	word_sim_search.py / word_sim_search.sh

		Running your queries!  Using the query matrix that was generated in embed_and_filter.py, each content word is queried for and searched in the annoy or faiss index (whichever one was created)

		Expected output files

		- eventprimitives.csv     //results in csv format of the retrieved sentences
		- wordsimsearch.stdout    //results in .txt format of the results





# Installing Dependencies...

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

### create and activate your virtual environment - you will need two for CPU processes (cpu_annoy, cpu_faiss) and two for GPU processes (gpu_annoy, gpu_faiss) depending on which indexing system you use.  Total of 4 virtenvs.

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
