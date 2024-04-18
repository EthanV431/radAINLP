from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def createTfidfEmbed(df):
    corpus = []
    for col in list(df.columns.values)[2:]:
        for cell in df[col]:
            corpus.append(cell)
    print("creating tf-idf embeddings...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
    list_v = vectorizer.fit_transform(corpus)
    print("tf-idf embeddings created")
    return list_v, vectorizer

def splitTfidf(list_v, vectorizer):
    ml_data_x = pd.DataFrame.sparse.from_spmatrix(list_v)
    new_cols = {}
    for i in range(100):
        new_cols[i] = vectorizer.get_feature_names_out()[i]
    ml_data_x.rename(mapper = new_cols, axis = 1, inplace = True)
    ml_data_y = []
    for i in range(954):
        ml_data_y.append(0)
    for i in range(954):
        ml_data_y.append(2)
    for i in range(954):
        ml_data_y.append(1)
    for i in range(954):
        ml_data_y.append(3)
    return train_test_split(ml_data_x, ml_data_y, test_size = 0.33, random_state = 1234)

def tfidfLR(x_train, x_test, y_train, y_test):
    lr_model = LogisticRegression(random_state = 123, max_iter = 3000)
    param_grid_ = {'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['sag', 'lbfgs', 'saga']}

    grid_search = GridSearchCV(lr_model, cv = 5, param_grid = param_grid_)
    print("determining optimal tf-idf training parameters...")
    grid_search.fit(x_train, y_train)
    
    print("training tf-idf logistic regression model...")
    tfidf_model = LogisticRegression(C = grid_search.best_params_['C'], solver = grid_search.best_params_['solver'], max_iter = 3000).fit(x_train, y_train)

    tfidf_lr_preds = tfidf_model.predict(x_test)
    tfidf_acc = accuracy_score(y_test, tfidf_lr_preds)
    print("tf-idf training complete")
    return tfidf_model, tfidf_acc