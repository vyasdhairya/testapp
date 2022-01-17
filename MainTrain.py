import pandas as pd
import string
from sklearn.metrics import f1_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import numpy as np
from nltk.corpus import stopwords

#load data
DATA_PATH1 = 'chromium.csv'
data1 = pd.read_csv(DATA_PATH1, sep='\t')
data_top1 = data1.head()
DATA_PATH2 = 'linux_bugs_usage_ready.csv'
data2 = pd.read_csv(DATA_PATH2, sep='\t')
data_top2 = data2.head()
#merge title and message
def merge_title_and_message(data, message_col_name='message'):
  data['text'] = data['title'] + ' ' + data[message_col_name]
  data = data.drop(['title'], axis=1)
  data = data.drop([message_col_name], axis=1)
  return data
data1 = merge_title_and_message(data1, message_col_name='description')
data2 = merge_title_and_message(data2)

def strip_punctuations(data, column_name='text'):
  '''
  Strips punctuations from the end of each token.
  This uses suggestion from https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
  to accomplish this really fast.
  '''
  translator = str.maketrans('', '', string.punctuation)
  data['text'] = data['text'].map(lambda s : str(s).translate(translator))
  return data
data1 = strip_punctuations(data1)
data2 = strip_punctuations(data2)

def remove_linux_garbage(data):
  '''
  Linux data contains lots of garbage, e.g. memory addresses - 0000f800
  '''
  def is_garbage(w):
    return len(w) >= 7 and sum(c.isdigit() for c in w) >= 2

  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if not is_garbage(w) else ' ', s.split())))
  return data
data2 = remove_linux_garbage(data2)

def cast_to_lowercase(data):
  data['text'] = data['text'].map(lambda s : s.lower())
  return data
data1 = cast_to_lowercase(data1)
data2 = cast_to_lowercase(data2)

def remove_stopwords(data):
  stop_words = stopwords.words('english')
  translator = str.maketrans('', '', string.punctuation)
  stop_words = set([w.translate(translator) for w in stop_words]) # Apostrophes were removed already
  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if w not in stop_words else ' ', s.split())))
  return data
data1 = remove_stopwords(data1)
data2 = remove_stopwords(data2)

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
data1 = remove_rare_words(data1)
data2 = remove_rare_words(data2)

from sklearn.ensemble import ExtraTreesClassifier
def essemble_classify(data,class_to_predict):
  extra_params={'min_df': 0.001}  
  data = shuffle(data, random_state=77)
  num_records = len(data)
  data_train = data
  data_test = data[int(0.85 * num_records):]

  train_data = [x[0] for x in data_train[['text']].to_records(index=False)]
  train_labels = [x[0] for x in data_train[[class_to_predict]].to_records(index=False)]

  test_data = [x[0] for x in data_test[['text']].to_records(index=False)]
  test_labels = [x[0] for x in data_test[[class_to_predict]].to_records(index=False)]

  # Create feature vectors 
  vectorizer = TfidfVectorizer(**extra_params)
  # Train the feature vectors
  train_vectors = vectorizer.fit_transform(train_data)
  test_vectors = vectorizer.transform(test_data)

  # Perform classification with SVM, kernel=linear 
  model = ExtraTreesClassifier(n_estimators=20, max_depth=None, min_samples_split=2, random_state=0)
  print('Training the model!')
  model.fit(train_vectors, train_labels) 
  train_prediction = model.predict(train_vectors)
  test_prediction = model.predict(test_vectors)

  train_accuracy = np.sum((np.array(train_labels) == np.array(train_prediction))) * 1.0 / len(train_labels)
  print('Training accuracy: ' + str(train_accuracy))

  test_accuracy = np.sum((np.array(test_labels) == np.array(test_prediction))) * 1.0 / len(test_labels)
  print('Test accuracy: ' + str(test_accuracy)) 
  print('F1 score: ' + str(f1_score(test_labels, test_prediction, average='weighted')))
  return model,vectorizer
model1,vectorizer1=essemble_classify(data1,'type')
import pickle
pickle.dump(model1, open('ET1.pkl', 'wb'))
pickle.dump(vectorizer1, open('VZ1.pkl', 'wb'))
import bz2
sfile = bz2.BZ2File('ET11', 'w')
pickle.dump(model1, sfile)

model2,vectorizer2=essemble_classify(data2,'importance')
pickle.dump(model2, open('ET2.pkl', 'wb'))
pickle.dump(vectorizer2, open('VZ2.pkl', 'wb'))
sfile = bz2.BZ2File('ET22', 'w')
pickle.dump(model2, sfile)



