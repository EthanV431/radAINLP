from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import os

def createNGram(df, n):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/NGramResults/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    for col in list(df.columns.values)[1:]:
        corpus = df[col]
        corpus = [item for item in corpus if not isinstance(item, int)]
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                        for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        top_n_bigrams=words_freq[:10][:10]
        x,y = map(list,zip(*top_n_bigrams))
        plot = sns.barplot(x=y,y=x)
        figure = plot.get_figure()
        print("Saving " + col + " ngram plot")
        figure.savefig(results_dir + col + "_ngrams.png", bbox_inches="tight") 
        plot.figure.clf()