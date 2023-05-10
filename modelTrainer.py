import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import dataHandler


# Creates the machine learning model and returns it. This method creates the model and the vectorizer and stores them
# locally on the computer, so this process should only happen once.
def createModel():
    trainX, testX, trainY, testY = dataHandler.getTrainTestSets()

    vectorAlgo = TfidfVectorizer()
    trainXVector = vectorAlgo.fit_transform(trainX)

    joblib.dump(vectorAlgo, 'Vectorizer.pkl')

    toxicPredictor = MultiOutputClassifier(RandomForestClassifier(n_estimators=125))
    toxicPredictor.fit(trainXVector, trainY)

    joblib.dump(toxicPredictor, 'ToxicCommentClassifier.pkl')

    return toxicPredictor


# Creates and returns a vectorizer for the model to use.
# NOTE: this should not need to be called as one is created and stored locally as Vectorizer.pkl, but it is here
# incase for some reason it was not created when the model was or was deleted.
def createVectorizer():
    trainX, testX, trainY, testY = dataHandler.getTrainTestSets()

    # Convert the comments to a numerical form so the algorithm can operate on them. The Term Frequency-Inverse Document
    # Frequency (TF-IDF) technique is used.
    vectorAlgo = TfidfVectorizer()
    vectorAlgo.fit_transform(trainX)
    vectorAlgo.transform(testX)
    joblib.dump(vectorAlgo, 'Vectorizer.pkl')

    return vectorAlgo
