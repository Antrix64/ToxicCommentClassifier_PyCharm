import pandas
import zipfile
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split


# The cleanData function preprocesses data to make it a consistent format for use in the machine learning model.
# It makes the text all lowercase, removes numbers and special characters, separate the data into words (tokenize),
# removes common words that are not meaningful, converts words to their root word (lemmatization), and puts the words
# back into their original sentences.
def cleanData(data):
    # Convert the data to lowercase.
    data = data.str.lower()
    # Remove special characters and numbers from the data.
    data = data.apply(lambda i: re.sub(r"[^a-zA-Z]", " ", i))
    # separate the words in the data.
    data = data.apply(word_tokenize)

    # Remove words that do not provide meaningful data.
    removeWords = stopwords.words("english")
    data = data.apply(lambda i: [word for word in i if word not in removeWords])

    # Convert words to their root form.
    rootWords = WordNetLemmatizer()
    data = data.apply(lambda i: [rootWords.lemmatize(word) for word in i])

    # Join the words back into sentences
    data = data.apply(lambda i: " ".join(i))

    # Return cleaned data.
    return data


# Returns data from a csv file within a zipfile as a pandas.read_csv object.
def loadCSVFromZipFile(zipFile, csvFile):
    # Extract the data from the CSV files within the zip file.
    with zipfile.ZipFile(zipFile, 'r') as trainZip:
        with trainZip.open(csvFile) as trainFile:
            csvData = pandas.read_csv(trainFile)

    return csvData


# Splits the data into training and test sets and returns train x, test x, train y, test y data.
def getTrainTestSets():
    # List of column headers for the toxicity labels
    toxicLabels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    data = loadCSVFromZipFile('toxic_subset_10901.zip', 'toxic_subset_10901.csv')
    dataY = data[toxicLabels]
    # Get cleaned data
    cleanedComments = cleanData(data['comment_text'])

    return train_test_split(cleanedComments, dataY, test_size=0.25, random_state=50)


