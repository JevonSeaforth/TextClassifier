# TextClassifier

## Brief Description
This project implements a natural language processing pipeline for classifying text documents into predefined categories using machine learning. The main goal of this project is to showcase how word embeddings, specifically Word2Vec, can be used in combination with neural networks to classify news articles as either "real" or "fake." This project helps illustrate the process of text classification, feature extraction, and model evaluation.

Originally written for a Natural Language Processing course. 
CSV file sourced from kaggle. Modified to fit size requirements for GitHub. 

## Key techniques: 

### Word2Vec Embedding
Word2Vec is a popular technique in NLP that transforms words into vectors, allowing the machine learning model to better understand the semantic relationships between words. By training a Word2Vec model on the text data, we convert individual words into continuous-valued vectors which capture the meaning of the words in context.

### Preprocessing
The text corpus is preprocessed to clean and tokenize the data. This includes:

  Removing stopwords, punctuation, and irrelevant tokens.
  Tokenizing the text into words.
  Applying lowercasing to standardize the text.
  
### Neural Network Classifier
The project utilizes a neural network classifier (MLPClassifier) to perform the classification task. Two variations of the neural network configurations are used:

  1 Hidden Layer, 100 Nodes: A simpler configuration that provides reasonable accuracy on the dataset.
  100 Hidden Layers, 10 Nodes: A more complex model which often leads to overfitting and reduces accuracy due to the excessive number of layers.
  The model's performance is evaluated using accuracy scores on both training and testing data.

### Training and Evaluation
  The dataset is split into training and testing subsets using a 90/10 split. The neural network is trained using the training data, and then the model is evaluated using the accuracy score metric    on both the training and testing datasets.
