from gensim.models import Word2Vec
import multiprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def createWordToVecEmbed(df):
    corpus = []
    for col in list(df.columns.values)[2:]:
        for cell in df[col]:
            corpus.append(cell)
    print("creating word2vec model...")
    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count = 5,window = 5,vector_size = 300,workers = cores-1,max_vocab_size = 100000)
    corpus = [c.split() for c in corpus]
    model.build_vocab(corpus)
    model.train(corpus,total_examples=model.corpus_count,epochs=50)
    print("word2vec model created")
    print("obtaining word2vec averages...")
    return model, corpus

def splitWordToVec(model, corpus):
    X = model.wv[model.wv.key_to_index]
    words = list(model.wv.key_to_index)
    lst = []
    for i in range(len(corpus)):
        temp = []
        for j in corpus[i]:
            if j in words:
                temp.append(X[words.index(j)])
            else:
                temp.append(np.zeros(300))
        lst.append(np.average(np.array(temp), axis=0))
    for i in range(len(lst)):
        if type(lst[i]) == np.float64:
            lst[i] = np.zeros(300)
    arr = np.array(lst)
    print("word2vec averages obtained")
    ml_data_y = []
    for i in range(954):
        ml_data_y.append(0)
    for i in range(954):
        ml_data_y.append(2)
    for i in range(954):
        ml_data_y.append(1)
    for i in range(954):
        ml_data_y.append(3)
    return train_test_split(arr, ml_data_y, test_size=0.33, random_state=1234)

def wordToVecLR(x_train, x_test, y_train, y_test):
    lr_model = LogisticRegression(random_state = 123, max_iter = 3000)
    param_grid_ = {'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['sag', 'lbfgs', 'saga']}

    print("determining optimal word2vec training parameters...")
    grid_search = GridSearchCV(lr_model, cv = 5, param_grid = param_grid_)
    grid_search.fit(x_train, y_train)
    
    print("training word2vec logistic regression model...")
    wordToVec_model = LogisticRegression(C = grid_search.best_params_['C'], solver = grid_search.best_params_['solver'], max_iter = 3000).fit(x_train, y_train)

    wordToVec_lr_preds=wordToVec_model.predict(x_test)
    
    wordToVec_acc = accuracy_score(y_test, wordToVec_lr_preds)
    print("word2vec model trained")
    return wordToVec_model, wordToVec_acc
