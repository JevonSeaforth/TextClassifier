import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

# file path
pathToCSV = 'news.csv'

# Note: I'm running the data through 2 different neural networks, one with 100 hidden layers/10 nodes and the other with 1 hidden layer, 100 nodes
# Based on the accuracy score we can see that the 100 hidden layers significantly reduces accuracy - most likely due to overfitting
# I have tested a few different permutations of neural networks but want to show the higher accuracy configuration vs lower accuracy configuration.
# Both results will be printed.

# Defined main function for readability
def main():
    # read the news.csv file into a pandas dataframe
    newsCorpus = pd.read_csv(pathToCSV)

    # Apply preprocessing to corpus
    listOfWords = preprocessCorpus(newsCorpus)

    # train word2vec with listOfWords.
    word2VecModel = Word2Vec(listOfWords, vector_size=100, window=5, min_count=1, workers=4)

    # consolidate vectors from word2vec model into input features 
    inputFeaturesForModel = consolidateVectors(listOfWords, word2VecModel)

    # Convert labels in news corpus from strings to numerical values
    newsCorpus['label'] = newsCorpus['label'].map({'REAL': 0, 'FAKE': 1})

    # store all of the numerical labels in y
    actualLabels = newsCorpus['label']

    # split input features and labels into training and testing
    # 90% training, 10% testing
    inputFeaturesTraining, inputFeaturesTesting, trainingLabels, testingLabels = train_test_split(inputFeaturesForModel, actualLabels, test_size=0.1, train_size=0.9, random_state=42)

    # set classifier values 
    alpha = 0.0001
    activationFunction='logistic'
    # was getting warning that classifier wasn't fully converging, up the iterations to 2000 to eliminate this issue
    maximumIterations=2000

    # Initializing the classifer on a single hidden layer, 100 nodes
    neuralNetClassifier = MLPClassifier(hidden_layer_sizes=(100,), activation=activationFunction, solver='adam', random_state=42, max_iter=maximumIterations, alpha=alpha)

    # train the neural network with the training input features, and the training labels.
    neuralNetClassifier.fit(inputFeaturesTraining, trainingLabels)

    # evaluate the accuracy the model on training features
    accuracyOfTraining = accuracy_score(trainingLabels, neuralNetClassifier.predict(inputFeaturesTraining))

    # evaluate the accuracy score on testing features
    accuracyOfTest = accuracy_score(testingLabels, neuralNetClassifier.predict(inputFeaturesTesting))

    # printing accuracy scores
    print("\nThe accuracy of the network using training data on 1 hidden layer, 100 nodes:", accuracyOfTraining)
    print("The accuracy of the network using testing data, 100 nodes:", accuracyOfTest)
    print("\n")
    # model likely underfitting with only 10 nodes in each layer
    # Specify an input layer of 10 nodes / 100 layers
    sizeOfLayer = tuple([10] * 100)

    # Initialize the classifer on a 10 nodes and 100  hidden layers
    secondMLP = MLPClassifier(hidden_layer_sizes=(sizeOfLayer), activation=activationFunction, solver='adam', random_state=42, max_iter= maximumIterations, alpha=alpha)
    
    # train the second model
    secondMLP.fit(inputFeaturesTraining, trainingLabels)

    # evaluate the accuracy the second model on training and testing data 
    accuracyOfTraining2 = accuracy_score(trainingLabels, secondMLP.predict(inputFeaturesTraining))
    accuracyOfTesting2 = accuracy_score(testingLabels, secondMLP.predict(inputFeaturesTesting))

    # print results of second model
    print("The accuracy of the network using training data on 100 hidden layers, and 10 nodes:", accuracyOfTraining2)
    print("The accuracy of the network using testing data on 100 hidden layers, and 10 nodes:", accuracyOfTesting2)

    print("\n NOTE: Model is significantly less accurate on 100 hidden layers and 10 nodes than 1 hidden layer with 100 nodes.")
    print("    Likely an issue with underfitting, as the dataset is probably too small for so many layers...")

# Definition: Preprocess the data so it is suitable for passing to Word2Vec
# Parameters: dataframe object
# Returns: list of processed data 
def preprocessCorpus(corpus):
    # Process the CSV data
    # Check to ensure 'text' and 'label' values in CSV are not missing values
    # If missing values, drop the row and change dataframe index so we don't access that row   
    corpus.dropna(subset=['text', 'label'], inplace=True)
    corpus.reset_index(drop=True, inplace=True) 

    # Store processed data
    processedDocuments = []
    stop_words = set(stopwords.words('english'))
    
    # iterate through each row in corpus
    for _, row in corpus.iterrows():
        # store title and text 
        document = ' '.join([str(row['title']), str(row['text'])]).strip()
        
        # If combined_text is an empty string, skip it
        if document == "":
            continue
        
        # tokenize the words in documenet
        tokenizedWords = word_tokenize(document.lower())
        
        # filter out stop words
        filteredStopWords = []
        for word in tokenizedWords: 
            if word not in stop_words:
                filteredStopWords.append(word)
        
        # add the filtered text to procecssedData
        processedDocuments.append(filteredStopWords)

    # return the corpus with proper preprocessing applied
    return processedDocuments

# Definition: Convert model returned from Word2Vec into list of input features, calculates the mean for each vector
# Parameters: dataframe object, and word2vec model
# Returns: list of processed data 
def consolidateVectors(listOfWords, Word2VecModel): 
    # Iterate through each word in pre-processed corpus 
    # For each word, check if the word is in the Word2Vec model
    # If it is, calculate mean of all values to consolidate into 1 number, then append to consolidatedWordsInModel
    # Else, skip, don't include
    consolidatedWordsInModel = []
    # Iterate through each document
    for doc in listOfWords:
        wordsInModel = []
        # For each word in document
        for word in doc:
            # check if word in model
            if word in Word2VecModel.wv:
                # if yes, append to list
                wordsInModel.append(Word2VecModel.wv[word])
        
        # now check if there are any elements in vector
        if wordsInModel != "":
            # calculate mean and add to consolidatedWordsInModel
            # note: mean is one option... I could use other methods to consolidate
            consolidatedWordsInModel.append(np.mean(wordsInModel, axis=0))

    # return list after all words have been visited.
    return consolidatedWordsInModel

if __name__ == "__main__":
    main()