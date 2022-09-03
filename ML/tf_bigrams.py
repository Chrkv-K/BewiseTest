import collections
import codecs
import re
import os

script_dir = os.path.dirname(__file__) 

file = open(os.path.join(script_dir, 'multi_turn.txt'), 'r', encoding='utf-8-sig')
tf_bigrams = codecs.open(os.path.join(script_dir, 'tf_bi_multi_turn.txt'), "w", "utf-8")

bigrams_array = []

# проходимся по каждой строке файла и делим ее на биграммы, если это возможно
# результат записываем в массив bigrams_array

for string in file:
    string = string.split()
    if len(string) >= 2:
        counter = 0
        while counter < (len(string) - 1):
           bigrams_array.append(str((string[counter]).lower()) + " " + str((string[counter+1]).lower()))
           counter += 1
    elif len(string) == 1:
        bigrams_array.append((string[0]).lower())
    else:
        continue

# подсчитываем частотность биграмм в массиве
counter = str(collections.Counter(bigrams_array))

# убираем лишние символы и записываем результат в файл
clean_counter = re.sub(r", '", "\n", counter)
clean_counter = re.sub(r"(\\n'|\')", "", clean_counter)
tf_bigrams.write(clean_counter)
file.close()
tf_bigrams.close()
