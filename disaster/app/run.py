import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from  plotly.graph_objs import Pie,Bar


app = Flask(__name__)

def tokenize(text):
    """
    Tokenzie text to words with word_tolenzie,lower,strip,WordNetLemmatizer
    :param text:the text for tokenize   
    :type text:string
    :return: a list contain tokenzied words
    :rtype:list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def return_figures(categories_names,categories_counts,genre_names,genre_counts):
    """Creates  plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """ 
    graph_one = []    
    graph_one.append(
      Bar(
      x = categories_names,
      y = categories_counts,
      )
    )

    layout_one = dict(title = 'Count of Message in Categories(Top 10)',
                xaxis = dict(title = 'Type'),
                yaxis = dict(title = 'Count'),
                )

   
    graph_two = []

    graph_two.append(
      Bar(
      x = genre_names,
      y = genre_counts,
      )
    )

    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'),
                )    
    
    raph_three = []    
    graph_three.append(
      Pie(
          labels = categories_names,
          values = categories_counts/categories_counts.sum(),
      )
    )
    layout_three = dict(title = 'Percentage of Messages in Categories(Top 10)',
                )
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    #figures.append(dict(data=graph_four, layout=layout_four))

    return figures

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Extract data needed for visuals
    :return:render_template
    :rtype:render_template render web page with plotly graphs
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories_counts = df.drop(columns=['id','message','genre','original']).sum().sort_values(ascending=False)[0:10]
    categories_names = list(categories_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = return_figures(categories_names,categories_counts,genre_names,genre_counts)
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    save user input in query
    :return:render_template
    :rtype:render_template render web page with plotly graphs
    """
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
