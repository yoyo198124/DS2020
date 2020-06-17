import sys
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ Load data from database
    Args:
	database_filepath(String):database file path
    Returns:
	X(Pandas dataframe):messages dataframe
        y(Pandas dataframe):category dataframe
        category_name(List):list of category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    X =  df.message
    y =  df.drop(['id','message','original','genre'],axis=1)   
    category_names = y.columns.values    
    return X,y,category_names

def tokenize(text):
    """Tokenzie text to words with word_tolenzie,lower,strip,WordNetLemmatizer
    Args:
	text(String):the text for tokenize   
    returns:
	stemmed(List): a list contain tokenzied words
    """
    #1 Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #2 Tokenize text
    words = word_tokenize(text)
    #3 Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    #4 # Lemmatize verbs by specifying pos
    lemmatizer = WordNetLemmatizer()
    stemmed = [ lemmatizer.lemmatize(w) for w in words]
    return stemmed


def build_model():
    """Build model
    Args:return:MultiOutputClassifierr(RandomForestClassifier()) model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 4],
    'clf__estimator__criterion': ['entropy']
     }
    # create grid search object
    model =  GridSearchCV(pipeline, param_grid=parameters,cv=2, n_jobs=-1, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
    :param model:model for evaluate
    :type model: model
    :param X_test:test input dataframe
    :type X_test:pd.dataframe
    :param Y_test:test output dataframe
    :type Y_test:pd.dataframe
    :param category_names:list of category names
    :type category_names:list
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save model to pkl file
    :param model:model for save
    :type model:model
    :param model_filepath:model file path
    :type model_filepath:string 
    """
    output = open(model_filepath, 'wb')
    s = pickle.dump(model, output)
    output.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'category_namestrain_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
