from metaflow import FlowSpec, step

class NLPFlow(FlowSpec):
        
    @step
    def start(self):
        from prep_data import prepareData
        self.df = prepareData()
        self.next(self.word_cloud)

    @step
    def word_cloud(self):
        from word_cloud import createWordClouds

        createWordClouds(self.df)
        self.next(self.n_gram)

    @step
    def n_gram(self):
        from n_gram import createNGram

        createNGram(self.df, 2)
        self.next(self.tfidf, self.word_to_vec)
    
    @step
    def tfidf(self):
        from tf_idf import createTfidfEmbed, splitTfidf, tfidfLR

        list_v, vectorizer = createTfidfEmbed(self.df)
        x_train, x_test, y_train, y_test = splitTfidf(list_v, vectorizer)
        self.tfidf_model, self.tfidf_acc = tfidfLR(x_train, x_test, y_train, y_test)
        self.next(self.join)

    @step
    def word_to_vec(self):
        from wordtovec import createWordToVecEmbed, splitWordToVec, wordToVecLR

        model, corpus = createWordToVecEmbed(self.df)
        x_train, x_test, y_train, y_test = splitWordToVec(model, corpus)
        self.wordToVec_model, self. wordToVec_acc = wordToVecLR(x_train, x_test, y_train, y_test)
        self.next(self.join)
    
    @step
    def join(self, inputs):
        print("comparing model performance...")
        if inputs.tfidf.tfidf_acc > inputs.word_to_vec.wordToVec_acc:
            self.final_model = inputs.tfidf.tfidf_model
            print("tf-idf model selected")
        else:
            self.final_model = inputs.word_to_vec.wordToVec_model
            print("word-to-vec model selected")
        self.next(self.end)

    @step
    def end(self):
        import joblib
        print("Saving selected model")
        joblib.dump(self.final_model , 'Results/final_model.pkl')

if __name__ == '__main__':
    NLPFlow()