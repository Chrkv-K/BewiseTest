from sklearn.metrics import recall_score
import pandas as pd
from sklearn import svm
import joblib
import gensim
import os

script_dir = os.path.dirname(__file__) 

fasttext_model = gensim.models.KeyedVectors.load(os.path.join(script_dir, '214/model.model'))
data = pd.read_csv(os.path.join(script_dir, 'dataset.csv'), sep=';', decimal=';')
words_for_model = (data.drop(['greetings', 'goodbye'], axis=1))

# для получения векторов, необходимых для обучения, используем модель fasttext_model
# если строка датасета состоит из одного слова, то получаем его вектор,
# если из двух, то находим сумму двух векторов слов
# вектора записываем в массив X_train

X_train = []
for index, row in words_for_model.iterrows():
    row = row['words'].split()
    if len(row) == 1:
        vector = fasttext_model[row[0]]
    else:
        vector = fasttext_model[row[0]] + fasttext_model[row[1]]
    X_train.append(vector)

# соответствующие теги для каждого вектора берем из колонок датасета dataset.csv
# greetings - для приветствий
# goodbye - для прощаний
# создаем бинарный классификатор используя метод опорных векторов
# тренируем и записываем модель

Y_train = data['greetings']
C = 500.0
svc = svm.SVC(C=C)
model = svc.fit(X_train, Y_train)
joblib.dump(model, os.path.join(script_dir, 'model_greetings_1.pkl'))

Y_train = data['goodbye']
C = 500.0
svc = svm.SVC(C=C)
model = svc.fit(X_train, Y_train)
joblib.dump(model, os.path.join(script_dir, 'model_goodbye_1.pkl'))