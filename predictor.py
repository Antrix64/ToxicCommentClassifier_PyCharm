import joblib
import pandas
import dataHandler
import modelTrainer
from io import StringIO
from sklearn.metrics import classification_report

model = None
vectorizer = None


# Returns the model. If the model is None, it attempts to load an existing model. If the model isn't found, a new model
# is created and saved locally.
def getModel():
    global model
    if model is None:
        try:
            model = joblib.load('ToxicCommentClassifier.pkl')
        except FileNotFoundError:
            model = modelTrainer.createModel()
    return model


# Returns the vectorizer. If the vectorizer is None, it attempts to load an existing vectorizer. If the vectorizer
# isn't found, a new vectorizer is created and saved locally.
def getVectorizer():
    global vectorizer
    if vectorizer is None:
        try:
            vectorizer = joblib.load('Vectorizer.pkl')
        except FileNotFoundError:
            vectorizer = modelTrainer.createVectorizer()
    return vectorizer


# Uses the model and vectorizer to make a prediction about whether a comment would be considered toxic and returns it.
def predict(comment):
    # Formats the comment to be converted to a pandas data object, so it works correctly with the
    # dataHandler.cleandata() function.
    testString = f'''comment_text
    {comment}'''
    ioStr = StringIO(testString)
    testData = pandas.read_csv(ioStr)
    cleanedComment = dataHandler.cleanData(testData['comment_text'])

    # Vectorize the cleaned comment.
    vectorizedData = getVectorizer().transform(cleanedComment)

    # Send the vectorized comment to the model to make a prediction.
    prediction = getModel().predict(vectorizedData)

    # Format the results of the prediction and return them.
    result = prediction[0]
    labels = ['toxic', 'severely toxic', 'obscene', 'threatening', 'insulting', 'identity hate']
    count = sum(prediction[0])
    response = f"The comment, \"{comment}\" is "
    toxicLabels = []

    for label, number in zip(labels, result):
        if number == 1:
            toxicLabels.append(label)

    toxicString = ""
    if len(toxicLabels) == 1:
        toxicString = toxicLabels[0]
    elif len(toxicLabels) == 2:
        toxicString = " and ".join(toxicLabels)
    elif len(toxicLabels) > 2:
        try:
            toxicString = ", ".join(toxicLabels[:-1]) + ", and " + toxicLabels[-1]
        except IndexError:
            pass

    if count == 0:
        response += "non-toxic."
    else:
        response += toxicString

    return response


def getClassificationReport():
    trainX, testX, trainY, testY = dataHandler.getTrainTestSets()
    testXVector = getVectorizer().transform(testX)
    predictY = getModel().predict(testXVector)
    label = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    report = classification_report(testY, predictY, target_names=label, output_dict=True)

    return report
