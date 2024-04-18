import nltk
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def createWordClouds(df):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/WordCloudResults/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    corpus=[]
    lem=nltk.wordnet.WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    for col in list(df.columns.values)[1:]:
        for cell in df[col]:
            words = [w for w in nltk.word_tokenize(str(cell)) if (w not in stop_words)]
            words = [lem.lemmatize(w) for w in words if len(w)>2]
            corpus.append(words)
        wordcloud = WordCloud(
            background_color='white',
            stopwords = stop_words,
            max_words = 100,
            max_font_size = 30,
            scale = 3,
            random_state = 1)

        wordcloud = wordcloud.generate(str(corpus))

        plt.axis('off')
        plt.imshow(wordcloud)
        print("Saving " + col + " wordcloud")
        plt.savefig(results_dir + col + "_wordcloud.png")
        corpus = []