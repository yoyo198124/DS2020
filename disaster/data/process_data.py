import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from messages and categories and merge them
    Args:
    	messages_filepath(String):the path of massages file
        categories_filepath(Strig):the path of categories file
    Returns:
	df(Pandas dataframe):dataframe of meaasges merge categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    """Clean the datafrom remove "-" and build new categores columns 
    Args:
	 df(Pandas Dataframe):dateframe for clean
    Returns:
	 df(Pandas Dataframe):cleaned dataframe
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])   
        categories[column] = pd.to_numeric(categories[column])  
    for column in categories:
        categories.loc[categories[column] >1] = 1
        categories.loc[categories[column] <0] = 0
    df.drop(columns='categories',inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Save data to database
    Args:
	df(Pandas dataframe):dataframe for save
        database_filename(String):database file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
