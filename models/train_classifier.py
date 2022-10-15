import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    return X, Y, Y.columns.values


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=8), n_jobs=-1))
    ])
    parameters = {
        "clf__estimator__n_estimators": [50, 100, 200],
        "clf__estimator__min_samples_split": [2, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print("Total accuracy:")
    print((Y_pred == Y_test).mean().mean())


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        nltk.download('punkt')
        nltk.download('stopwords')

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
