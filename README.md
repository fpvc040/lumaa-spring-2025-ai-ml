# Movie Recommendation System

This project implements a content-based movie recommendation system that combines movie plot summaries with metadata (genres, languages, and countries) to generate movie suggestions based on user input.
1. The code preprocesses the data and ensures we have both the plot summary and attributes for the movie. 
2. It computes the features for the movie summaries and caches them for faster retrieval
3. It computes the features for various metadata attributes and caches them
4. It computes the cosine similarities and returns the top N matches. 

### Note: The first time, the pipeline caches the embeddings, but it may take some time. The embeddings have been processed in batches, and there are live progress bars. 

## Video Output Example
The example shows the change in video recommendations using a similar prompt for action genre but in different locations:  one space and intergalactic themed, and the other ocean and island themed. 
[![Live recommendations using python recommender system](https://drive.google.com/file/d/1aoYqxBHPjYmD4rqT7ygYioYNpPC5bfJN/view?usp=sharing)](https://drive.google.com/file/d/1aoYqxBHPjYmD4rqT7ygYioYNpPC5bfJN/view?usp=sharing)

## Requirements

To run this recommendation system, you need to have the following Python packages installed:

- `pandas`
- `numpy`
- `sentence-transformers`
- `scikit-learn`
- `argparse`

You can install them via pip:

```bash
pip install pandas numpy sentence-transformers scikit-learn argparse
```
or 
```bash
pip install -r requirements.txt
```
The data has been included in the github repo, but you can also download the data from [https://www.cs.cmu.edu/~ark/personas/](https://www.cs.cmu.edu/~ark/personas/). Unzip into the root project directory. 

# Examples

You can cd/ in to the src directory and simply call 
```bash
python main.py --query "Example Prompt"
```
An example prompt tested here is: 
```bash
 python main.py --query "I love English movies that have sci-fi elements, talk about the invention of a new science theorem, and are historically based."
```

The outcome is as follows: 
```bash
python main.py --query "I love English movies that have sci-fi elements, talk about the invention of a new science theorem, and are historically based."
Loading and processing data...
Finding recommendations...
Movie ID: 12073433
Movie Name: The Beginning or the End
Similarity: 0.3687
Metadata: {'Wikipedia Movie ID': '12073433', 'Plot': 'The film dramatizes the creation of the atomic bomb in the Manhattan Project and the subsequent bombing of Japan.', 'Freebase Movie ID': '/m/02vnx_8', 'Movie Name': 'The Beginning or the End', 'Release Date': '1947-02-19', 'Box Office Revenue': nan, 'Runtime': 112.0, 'Languages': '{"/m/02h40lc": "English Language"}', 'Countries': '{"/m/09c7w0": "United States of America"}', 'Genres': '{"/m/02l7c8": "Romance Film", "/m/07s9rl0": "Drama", "/m/01g6gs": "Black-and-white", "/m/082gq": "War film"}'}
--------------------------------------------------
Movie ID: 26213151
Movie Name: Remote Control
Similarity: 0.3662
Metadata: {'Wikipedia Movie ID': '26213151', 'Plot': "A video store clerk stumbles onto an alien plot to take over earth by brainwashing people with a bad '50s science fiction movie. He and his friends race to stop the aliens before the tapes can be distributed world-wide.", 'Freebase Movie ID': '/m/0b77wr3', 'Movie Name': 'Remote Control', 'Release Date': '1988-04-07', 'Box Office Revenue': nan, 'Runtime': 88.0, 'Languages': '{"/m/02h40lc": "English Language"}', 'Countries': '{"/m/09c7w0": "United States of America"}', 'Genres': '{"/m/05p553": "Comedy film", "/m/03npn": "Horror", "/m/06n90": "Science Fiction"}'}
--------------------------------------------------
Movie ID: 19165692
Movie Name: The Man Who Wouldn't Talk
Similarity: 0.3639
Metadata: {'Wikipedia Movie ID': '19165692', 'Plot': 'A courtroom drama, it sees an American scientist charged for murder by the British police for his supposed role in the death of an Eastern Bloc defector.', 'Freebase Movie ID': '/m/04ljctk', 'Movie Name': "The Man Who Wouldn't Talk", 'Release Date': '1958-01-21', 'Box Office Revenue': nan, 'Runtime': 91.0, 'Languages': '{"/m/02h40lc": "English Language"}', 'Countries': '{"/m/07ssc": "United Kingdom"}', 'Genres': '{"/m/0lsxr": "Crime Fiction", "/m/07s9rl0": "Drama"}'}
--------------------------------------------------
Movie ID: 2042539
Movie Name: The District!
Similarity: 0.3596
Metadata: {'Wikipedia Movie ID': '2042539', 'Plot': "The film displays the Hungarian, Roma, Chinese and Arab dwellers and their alliances and conflicts in a humorous way, embedded into a fictive story of a few schoolchildren's oil-making time-travel and a Romeo and Juliet-type love of a Roma guy towards a white girl.", 'Freebase Movie ID': '/m/06h2sy', 'Movie Name': 'The District!', 'Release Date': '2004-12-09', 'Box Office Revenue': nan, 'Runtime': 87.0, 'Languages': '{"/m/012psb": "Romani language", "/m/02h40lc": "English Language", "/m/02ztjwg": "Hungarian language"}', 'Countries': '{"/m/03gj2": "Hungary"}', 'Genres': '{"/m/01z4y": "Comedy", "/m/03q4nz": "World cinema", "/m/0hcr": "Animation"}'}
--------------------------------------------------
Movie ID: 17945020
Movie Name: Simon
Similarity: 0.3509
Metadata: {'Wikipedia Movie ID': '17945020', 'Plot': 'The Institute for Advanced Concepts, a group of scientists with an unlimited budget and a propensity for elaborate pranks, brainwash a psychology professor named Simon Mendelssohn who was abandoned at birth and manage to convince him, and the rest of the world, that he is of extraterrestrial origin. Simon escapes and attempts to reform American culture by overriding TV signals with a high power TV transmitter, becoming a national celebrity in the process.', 'Freebase Movie ID': '/m/047tdq7', 'Movie Name': 'Simon', 'Release Date': '1980-02', 'Box Office Revenue': 6000000.0, 'Runtime': 97.0, 'Languages': '{"/m/02h40lc": "English Language"}', 'Countries': '{"/m/09c7w0": "United States of America"}', 'Genres': '{"/m/06n90": "Science Fiction", "/m/06nbt": "Satire", "/m/01z4y": "Comedy"}'}
--------------------------------------------------
```

To regenerate the cache/embeddings, just delete the embeddings.npz and metadata_embeddings.npz. 

# How It Works
The system uses two models from the sentence-transformers library:

all-MiniLM-L6-v2 for encoding movie plot summaries into embeddings.
paraphrase-distilroberta-base-v1 for encoding movie metadata (Genres, Languages, Countries) into embeddings.

I chose those 2 models according to the task: all-MiniLM model is better at leanring longer semantic and contextual info, while the paraphrase is good for specific features such as language, genre. 


Given a user query (a movie plot or description), the system computes the similarity between the query and movie plots as well as the metadata. The recommendation system then returns the top N most similar movies based on both plot and metadata similarities.

# Data Structure
MovieSummaries/ - Folder for movie data
MovieSummaries/movie_summaries.txt — Contains the movie plot summaries.
MovieSummaries/movie.metadata.tsv — Contains the metadata of movies including genres, languages, and countries.
