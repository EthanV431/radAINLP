import pandas as pd
import nltk

def prepareData():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    df = pd.read_csv('open_ave_data.csv')
    for col in list(df.columns.values)[1:]:
        df[col] = df[col].fillna("")
    return df