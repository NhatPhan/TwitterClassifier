import os
import time
import string
import pickle
import csv
import pandas as pd

from operator import itemgetter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

from preprocessor import Preprocessor


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg
    

def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.


    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.


    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.


    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """
    
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """    
        if isinstance(classifier, type):
            classifier = classifier()
            
        model = Pipeline([
                    ('preprocessor', Preprocessor()),
                    ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
                    ('classifier', classifier)
                ])
        
        model.fit(X, y)
        return model

    # Label encode the labels
    labels = LabelEncoder()
    y = labels.fit_transform(y)
    
    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model = build(classifier, X_train, y_train)
    
    if verbose: print("Classification Report:\n")
    
    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))
    
    if verbose: print("Building complete model and saving")
    model = build(classifier, X, y)
    model.labels_ = labels
    
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def predict(model, text, n=20):

    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer then predict the given text
    This function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    
    # Check to make sure we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.name        
            )
        )
    
    tvec = model.transform(text)
    
     # Zip the feature names with the coefs and sort

    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    #topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []
    output.append("\n")
    
    # If text, add the predicted value to the output.
    predict_value = model.labels_.classes_[model.predict(text)]
    output.append("tweet is classified as: {}".format(predict_value))
    output.append("")
    
    # Create two columns with most negative and most positive features.

    #for (cp, fnp), (cn, fnn) in topn:
     #   output.append(
      #      "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
       # )

    return "\n".join(output)
        

if __name__ == "__main__":
    
    PATH = "model.pickle"
    
    if not os.path.exists(PATH):
        df = pd.read_csv("training_tweets/categorized_tweets.csv")
        X = df.Text
        y = df.Category
        model = build_and_evaluate(X, y, outpath=PATH)
    else:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
    
    text = ["Manchester is red!"]
    print(predict(model, text, 20))       