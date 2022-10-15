import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Loads csv files for messages and categories, merging them together
        Arguments:
            messages_filepath: filepath for messages csv file
            categories_filepath: filepath for categories csv file
        Returns:
            Merged dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, left_on="id", right_on="id")
    return df


def clean_data(df):
    """
        Cleans the dataframe and converts categories into separate columns
        Arguments:
            df: pandas dataframe
        Returns:
            Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    categories.columns = row.apply(lambda x: x[:-2])
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df.drop(labels=["categories"], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # The column related has some wrong values!
    # All values are supposed to be binary, the value 2 is clearly wrong.
    df.drop(df[df.related == 2].index, inplace=True)
    # drop duplicates
    df = df.drop_duplicates()
    assert (df.shape[0] - df.drop_duplicates().shape[0]) == 0
    return df


def save_data(df, database_filename):
    """
        Saves the dataframe in a SQL database
        Arguments:
            df: pandas dataframe
            database_filename: filename for SQL database
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("Messages", engine, index=False, if_exists="replace")


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
