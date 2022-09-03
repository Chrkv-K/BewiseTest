# импорт необходимых библиотек

import gensim
import pandas as pd
import joblib
import pymorphy2
import re
import pandas as pd
from yargy import rule, and_, Parser, or_
from yargy.predicates import gram
from yargy.interpretation import fact
from yargy.relations import gnc_relation
from yargy.pipelines import morph_pipeline
import nltk
from nltk.corpus import stopwords
import spacy
import os

def upper_replace(match):

    """
    Функция возвращает результат поиска регулярного выражения в верхнем регистре
    """

    return match.group(0).upper()

def search_using_models(classifer, text):

    """
    Функция классификации переданной строки, с использованием модели классификатора
    """

    """
    Функция делит строку на биграммы (если строка состоит из одного слова, то не делит).
    Затем для каждой биграммы (или слова) при помощи fasttext_model находится сумма векторов слов.
    Полученный вектор сравнивается с векторами биграмм (или слов) из датасета(dataset.csv),
    на котором были обучены модели classifer (classifer_goodbye и classifer_greetings),
    если моделью в строке находится хотя бы одно совпадение признака (приветствия/прощания),
    мы делаем вывод, что искомый паттерн найден
    """

    final_result = []
    bigrams_array_phrase = []
    phrase = text.split()
    if len(phrase) >= 2:
        counter = 0
        while counter < (len(phrase) - 1):
            bigrams_array_phrase.append(str((phrase[counter]).lower()) + " " + str((phrase[counter+1]).lower()))
            counter += 1
    elif len(phrase) == 1:
        bigrams_array_phrase.append((phrase[0]).lower())
    for element in bigrams_array_phrase:
        X = []
        element = element.split()
        if len(element) == 1:
            vector = fasttext_model[element[0]]
        else:
            vector = fasttext_model[element[0]] + fasttext_model[element[1]]
        X.append(vector)
        result = classifer.predict(X)
        if result == [0]:
            final_result.append(0)
        elif result == [1]:
            final_result.append(1)
    return final_result

nltk.download('stopwords')
stop_words_for_work = stopwords.words('russian')
stop_words_for_work.append('диджитал') # добавляем в список стоп-слов "Диджитал", которое может быть распознано как имя
nlp = spacy.load("ru_core_news_sm")
morph = pymorphy2.MorphAnalyzer()
script_dir = os.path.dirname(__file__) 

# модель fasttext_model была взята отсюда http://vectors.nlpl.eu/repository/20/214.zip
fasttext_model = gensim.models.KeyedVectors.load(os.path.join(script_dir, 'ML/214/model.model'))

# модели classifer_greetings и classifer_goodbye обучила самостоятельно
classifer_greetings = joblib.load(os.path.join(script_dir, 'ML/model_greetings.pkl'))
classifer_goodbye = joblib.load(os.path.join(script_dir, 'ML/model_goodbye.pkl'))

# словари для фиксирования наличия приветствия и прощания в диалогах
requirements_greeting = {}
requirements_goodbye = {}

gnc = gnc_relation()  # согласование по падежу, числу и роду для поиска имен

# шаблоны для поиска имени
Name = fact('Name',['first', 'middle', 'last'])

FIRST = gram('Name').interpretation(Name.first.inflected()).match(gnc)
MIDDLE = gram('Patr').interpretation(Name.middle.inflected()).match(gnc)
LAST = gram('Surn').interpretation(Name.last.inflected()).match(gnc)

# расписаны возможные вариации с именем (first), отчеством (middle) и фамилией (last)
NAME = or_(
    rule(FIRST, LAST),
    rule(LAST, FIRST),
    rule(FIRST),
    rule(LAST),
    rule(FIRST, MIDDLE, LAST),
    rule(LAST, FIRST, MIDDLE),
    rule(FIRST, MIDDLE)).interpretation(Name)

parser = Parser(NAME)

# работаем с файлом test_data.csv и добавляем в него колонку 'аdditional Information' для фиксирования найденной информации
data = pd.read_csv(os.path.join(script_dir, 'test_data.csv'), sep=',', encoding="utf-8-sig")
data.insert(data.shape[1], 'аdditional Information', "", True)
data.iterrows()

# цикл для поиска именованных сущностей
for i, row in data.iterrows():
    index = i
    name = ''

    # поиск имени
    # для ускорения работы скрипта, рассматриваем только первые строки фраз менеджера,
    # где с большей вероятностью он назовет свое имя
    if row['role'] == "manager" and row['line_n'] < 4: 
        text = (row['text']).split()
        # удаляем стоп-слова, чтобы повысить качество поиска
        string_without_sw = [word for word in text if not word.casefold() in stop_words_for_work]
        string_without_sw = " ".join(string_without_sw)
        for match in parser.findall(string_without_sw):
            # найденные варианты собираем в переменную и записываем в таблицу с тегом 'name'
            if not (match.fact.first is None):
                name += f"{match.fact.first}"
            if not (match.fact.middle is None):
                name += f" {match.fact.first}"
            if not (match.fact.last is None):
                name += f" {match.fact.first}"
            data.loc[data.index[index], 'аdditional Information'] = f"name=\'{name}\'"

    # поиск названия компании
    # результат работы модели на русском языке показало недостаточно выское качество,
    # чтобы его повысить, сокращаем варианты строк для поиска регулярными выражениями
    check = re.search(r"\bкомпани\w\b\s\w+\s\w+\b", str(row['text']))
    if check:
        text = re.sub(r'(?<=(\bкомпани\w\b\s))\w', upper_replace, str(row['text']))
        piece_of_text = re.findall(r"\bкомпани\w\b\s\w+\s\w+\b", text)
        piece_of_text = (piece_of_text[0]).split()
        string_without_sw = [word for word in piece_of_text if not word.casefold() in stopwords.words('russian')]
        string_without_sw = " ".join(string_without_sw)
        piece_of_text = nlp(string_without_sw)
        for ent in piece_of_text.ents:
            # найденные варианты записываем в таблицу с тегом 'organization'
            if ent.label_ == "ORG":
                if data.loc[data.index[index], 'аdditional Information'] == '':
                    data.loc[data.index[index], 'аdditional Information'] += f"organization=\'{ent}\'"
                else:
                    data.loc[data.index[index], 'аdditional Information'] += f"/organization=\'{ent}\'"

# проверяем наличие приветствий во фразах менеджера при помощи модели classifer_greetings
# для ускорения работы скрипта, рассматриваем только первые строки фраз менеджера,
# где с большей вероятностью он поздоровается

num_of_dialogs = data['dlg_id'].unique()
for dialog in num_of_dialogs:
    for i, row in data.iterrows():
        index = i
        if row['dlg_id'] == dialog and row['line_n'] < 4 and row['role'] == "manager":
            final_result = search_using_models(classifer_greetings, row['text'])
            if sum(final_result) > 0:
                # успешный поиск приветствий фиксируем в таблице с диалогами под тегом greeting
                # и в словаре requirements_greeting
                requirements_greeting[dialog] = "True"
                if data.loc[data.index[index], 'аdditional Information'] == '':
                    data.loc[data.index[index], 'аdditional Information'] += f"greeting=True"
                else:
                    data.loc[data.index[index], 'аdditional Information'] += f"/greeting=True"

# проверяем наличие прощания во фразах менеджера при помощи модели classifer_goodbye
# для ускорения работы скрипта, рассматриваем только последние строки диалога (только у менеджера),
# где с большей вероятностью он должен был попрощаться

for dialog in reversed(num_of_dialogs):
    counter_for_goodbye = 0
    for i, row in data.reindex(index=data.index[::-1]).iterrows():
        index = i
        if row['dlg_id'] == dialog and row['role'] == "manager" and counter_for_goodbye < 2:
            final_result = search_using_models(classifer_goodbye, row['text'])
            # успешный поиск прощания фиксируем в таблице с диалогами под тегом goodbye
            # и в словаре requirements_goodbye
            if sum(final_result) > 0:
                requirements_goodbye[dialog] = "True"
                if data.loc[data.index[index], 'аdditional Information'] == '':
                    data.loc[data.index[index], 'аdditional Information'] += f"goodbye=True"
                else:
                    data.loc[data.index[index], 'аdditional Information'] += f"/goodbye=True"
# записываем обновленный вариант таблицы
data.to_csv(os.path.join(script_dir, 'data_final.csv'))

# проверяем соответствие требований к менеджеру: обязательно поприветствовать клиента и попрощаться с ним
for num in num_of_dialogs: 
    if requirements_greeting.setdefault(num) == "True" and requirements_goodbye.setdefault(num) == "True":
        print(f'В диалоге {num} менеджер соответствует требованиям')