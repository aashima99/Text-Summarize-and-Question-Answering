# Text summarization and Question Answering system

## PROBLEM STATEMENT AND ITS SOLUTION
In a recent advance, the significance of text summarization accomplishes more attention due to data inundation on the web.  Hence, this information overwhelms yields in the 
big requirement for more reliable and capable progressive text summarizers. Text Summarization gains its importance due to its various types of applications just like the summaries of books, digest- (summary of stories), the stock market, news, highlights- (meeting, event, sport), Abstract of scientific papers, newspaper articles, magazine etc. The main advantage of a text summarization is reading time of the user can be reduced.  
The objective of our automatic text summarization system for Covid-19 data and Question Answering is to condense the origin text into a precise version preserves its report content and global denotation in order to provide relevant information to the users and also further solving their queries about Covid-19 in order to avoid confusion.  

## INTRODUCTION
An extractive summarization technique consists of selecting vital sentences, paragraphs, etc, from the original manuscript and concatenating them into a shorter form.  The significance of sentences is  strongly  based  on  statistical  and  linguistic features of sentences.  This document covers the extensive methodologies fitted, issues launch, exploration and future directions in text  summarization in our project. It covers the basic workflow of the project which includes the features we used for extractive text summarization and further we have also used BERT Summarizer for abstractive summarization of the Covid-19 data. For the question answering system we have used allennlp-models available on PyPI.

## IMPLEMENTATION AND BASIC WORKFLOW
The implementation will be done in a series of steps as follows:
1) Preprocessing: Preprocessing involves analyzing the input text based on the following parameters: a. Sentence count b. Sentence segmentation c. Word Stemming 
2) Sentence scoring: All the sentences are scored based on various features such as its similarity to title, presence on numerical data in the sentence, its format (bold, italics), length of sentences, presence of certain phrases and its word similarity. 

The features used are:
1) Cue Phrases
2) Numerical Data
3) Sentence Length
4) Sentence Position
5) Term Weight
6) Upper Case
7) Number of proper Noun
8) Title Features
9) Centrality
10) Thematic Features
11) Named Entity Recoginition

### BERT SUMMARIZER
We have also implemented a text summarizer using BERT that can summarize large data using just a few lines of code. 
The input format of BERTSUM is different when compared to the original model. Here, a [CLS] token is added at the start of each sentence in order to separate multiple sentences and to collect features of the preceding sentence. There is also a difference in segment embeddings. In the case of BERTSUM, each sentence is assigned an embedding of Ea or Eb depending on whether the sentence is even or odd. If the sequence is [s1,s2,s3] then the segment embeddings are [Ea, Eb, Ea]. This way, all sentences are embedded and sent into further layers. 
BERTSUM assigns scores to each sentence that represents how much value that sentence adds to the overall document. So, [s1,s2,s3] is assigned [score1, score2, score3]. The sentences with the highest scores are then collected and rearranged to give the overall summary of the article. 
BERTSUM has an in-built module called summarizer that takes in our data, accesses it and provided the summary within seconds.

### ALLENNLP FOR QUESTION ANSWER GENERATION
AllenNLP makes it easy to design and evaluate new deep learning models for nearly any NLP problem. AllenNLP makes it easy to design and evaluate new deep learning models for nearly any NLP problem, along with the infrastructure to easily run them in the cloud or on your laptop.


## WORKFLOW OF THE PROJECT 
![flowchart](https://github.com/infinity1013/TextSummarization/blob/main/static/flowchart.png)


### LIBRARIES USED

#### NLTK 
Natural Language Toolkit (NLTK) is a text processing library that is widely used in Natural Language Processing (NLP). It supports the high-performance functions of tokenization, parsing, classification, etc. The NLTK team initially released it in 2001 (Nltk.org, 2018). 

#### Scikit-learn 
Scikit-learn is a machine learning library in Python. It performs easy-to-use dimensional reduction methods such as Principal Component Analysis (PCA), clustering methods such as k- 10 means, regression algorithms such as logistic regression, and classification algorithms such as random forests (Scikit-learn.org, 2018). 

#### Pandas 
Pandas provides a flexible platform for handling data in a data frame. It contains many open-source data analysis tools written in Python, such as the methods to check missing data, merge data frames, and reshape data structure, etc. (“Pandas”, n.d.). 

#### Gensim 
Gensim is a Python library that achieves the topic modeling. It can process a raw text data and discover the semantic structure of input text data by using some efficient algorithms, such as Tf-Idf and Latent Dirichlet Allocation (Rehurek, 2009). 

#### Flask 
Flask, issued in mid-2010 and developed by Armin Ronacher, is a robust web framework for Python. Flask provides libraries and tools to build primarily simple and small web applications with one or two functions (Das., 2017). 

#### Bootstrap 
Bootstrap is an open-source JavaScript and CSS framework that can be used as a basis to develop web applications. Bootstrap has a collection of CSS classes that can be directly used to create effects and actions for web elements. Twitter’s team developed it in 2011 (“Introduction”, n.d.).


## Applications of Our Project:
1. People need to learn much from texts. But they tend to want to spend less time while doing this.
2. It aims to solve this problem by supplying them the summaries of the Covid-19 data from which they want to gain information.
3. Goals of this project are that these summaries will be as important as possible in the aspect of the texts’ intention.
4. The user would easily get answers to his queries.
5. Supplying the user, a smooth and clear interface.
6. Configuring a fast replying server system.


### Web Interface

#### Dashboard
![Alt Text](/images/Dashboard.png)


#### About the Project
![Alt Text](/images/About_Project.png)


#### Summarize
![Alt Text](/images/Summarize.png)


#### Question Answering
![Alt Text](/images/Q&A.png)


#### Analyse
![Alt Text](/images/Analyse.png)
