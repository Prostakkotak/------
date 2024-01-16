from gensim.models import Word2Vec
import re
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

w2v_model = Word2Vec.load('resources/vaskovsky.model')

russian_stopwords = stopwords.words("russian")

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

word_vectors = []
for j in range(1, 31):
    result = 0
    words = 0
    with open(f"resources/articles/{j}.txt", 'r', encoding='utf8') as file:
        for line in file:
            line = re.sub(patterns, ' ', line)
            for word in line.split():
                if word not in russian_stopwords and w2v_model.wv.has_index_for(word):
                    word_vectors.append(w2v_model.wv.get_vector(word))

sorted_vectors = sorted(word_vectors, key=lambda x: sum(x))

# Разделение на категории
category1 = sorted_vectors[:len(sorted_vectors)//3]
category2 = sorted_vectors[len(sorted_vectors)//3:2*len(sorted_vectors)//3]
category3 = sorted_vectors[2*len(sorted_vectors)//3:]

# Среднее арифметическое для каждой категории
avg_category1 = sum(category1) / len(category1)
avg_category2 = sum(category2) / len(category2)
avg_category3 = sum(category3) / len(category3)

print(avg_category1.sum())
print(avg_category2.sum())
print(avg_category3.sum())

# Набор ближайших по вектору слов
similar_words1 = w2v_model.wv.similar_by_vector(vector=avg_category1, topn=5)
similar_words2 = w2v_model.wv.similar_by_vector(vector=avg_category2, topn=5)
similar_words3 = w2v_model.wv.similar_by_vector(vector=avg_category3, topn=5)

print(similar_words1)
print(similar_words2)
print(similar_words3)