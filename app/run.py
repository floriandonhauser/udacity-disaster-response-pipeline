import json

import joblib
import nltk
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # data for messages by genre
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100 * gen_count / gen_count.sum(), 2)
    gen = list(gen_count.index)

    # data for messages by categories
    categories = df[df.columns[4:]]
    cate_counts = (categories.mean() * categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)

    # create visuals
    graphs = [
        {
            "data": [
                {
                    "type": "pie",
                    "hole": 0.2,
                    "domain": {
                        "x": gen_per,
                        "y": gen
                    },
                    "marker": {
                        "colors": [
                            "#00B8A9",
                            "#F6416C",
                            "#FFDE7D"
                        ]
                    },
                    "textinfo": "label+value",
                    "labels": gen,
                    "values": gen_count
                }
            ],
            "layout": {
                "title": "Messages by Genre"
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "x": cate_names,
                "y": cate_counts,
                "marker": {
                  "color": "#00B8A9"}
                }
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'margin': {
                    'b': 200
                }
            },
            'color': "#00B8A9"
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    nltk.download('omw-1.4')
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
