
def prepro1(df1):
    import pandas as pd
    import string
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    #load data
    txt=df1
    datau = {'text':[txt]}
    df1 = pd.DataFrame(datau)


    
    def strip_punctuations(data, column_name='text'):
      '''
      Strips punctuations from the end of each token.
      This uses suggestion from https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
      to accomplish this really fast.
      '''
      translator = str.maketrans('', '', string.punctuation)
      data['text'] = data['text'].map(lambda s : str(s).translate(translator))
      return data
    df2 = strip_punctuations(df1)
    
    def remove_linux_garbage(data):
      '''
      Linux data contains lots of garbage, e.g. memory addresses - 0000f800
      '''
      def is_garbage(w):
        return len(w) >= 7 and sum(c.isdigit() for c in w) >= 2
    
      data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if not is_garbage(w) else ' ', s.split())))
      return data
    #df3 = remove_linux_garbage(df2)
    
    def cast_to_lowercase(data):
      data['text'] = data['text'].map(lambda s : s.lower())
      return data
    df3 = cast_to_lowercase(df2)
    
    def remove_stopwords(data):
      stop_words = stopwords.words('english')
      translator = str.maketrans('', '', string.punctuation)
      stop_words = set([w.translate(translator) for w in stop_words]) # Apostrophes were removed already
      data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if w not in stop_words else ' ', s.split())))
      return data
    df4 = remove_stopwords(df3)
    
    def remove_rare_words(data, min_count=3):
      wc = {} # WordCount
      def proc_word(s):
        for w in set(s.split()):
          if w in wc:
            wc[w] += 1
          else:
            wc[w] = 1
      for index, row in data.iterrows():
        proc_word(row['text'])
      data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if wc[w] >= min_count else ' ', s.split())))
      return data
    #df5 = remove_rare_words(df4)
    test_data = [x[0] for x in df4[['text']].to_records(index=False)]
    import pickle
    vectorizer1=pickle.load(open('VZ1.pkl', 'rb'))
    test_vector = vectorizer1.transform(test_data)
    import bz2
    sfile = bz2.BZ2File('ET11', 'r')
    model1=pickle.load(sfile)
    test_prediction = model1.predict(test_vector)
    return test_prediction

