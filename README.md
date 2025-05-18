### Overview

In this repo, we'll implement multiple recommendation system algorithms from scratch, then validate them on the dataset in the data folder. Here's the specification of what we're going to implement:

#### List of implemented algorithms:
1. Apriori
2. Hashtree
3. Collaborative filtering (item-based or user-based)
4. Alternating Least Squares (ALS) / SVD++

For no.3 and no.4 algorithm, we'll have to create embedding for text data. Here's the embedding scheme that we'll implement:
- Movie genres: one-hot encoding
- User tags for movie: GLoVe or BM25

A movie will be embedded by concatenation of the following features:

- Reduced PCA representation of genre ohe
- Average of user-tag embeddings (Honestly I'm not sure about this part: there are tags that differs in semantics in a very delicate way, and there are tags that is a phrase, not a single word. Plus we are discouraged to use neuron-based embedding models here).

### Dataset structure

The data folder contains the following csv files:
- genome_scores: comprises of three columns
+ movieId: id of a movie
+ tagId: id of a tag (given in genome_tags.csv file)
+ relevance: relevance score of a movie to a tag

- genome_tags: comprises of two columns
+ tagId: id of the tag
+ tag: tag content (could be a phrase or a single word)

- link: this csv can be ignored

- movie: comprises of three columns
+ movieId: id of a movie
+ title: title of a movie
+ genres: genres of a movie, multiple genres are seperated by | (for example, "Adventure|Animation|Children|Comedy|Fantasy")

- rating: comprises of four columns
+ userId: id of a user
+ movieId: id of a movie
+ rating: rating of a movie (on a range of 0.5 to 5.0)
+ timestamp: dunno what this means (but we don't use this column)

- tag: comprises of three columns (these tags are different from those in genome_tags file, but we'll concatenate them later on)
+ userId: id of a user
+ movieId: id of a movie
+ tag: text tag that the user gives the movie (can be a phrase or a single word)
