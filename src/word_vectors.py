import string

import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例语料库
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?'
]


# 预处理文本：小写化和去除标点符号
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)


# for doc in corpus:
    # print(preprocess(doc))

# 对每个文档进行预处理
processed_corpus = [preprocess(doc) for doc in corpus]

# 训练Word2Vec模型 向量大小为100维
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=1, sg=0)

# 获取词汇表中的所有词
vocab = model.wv.key_to_index
# print(vocab)

# 创建一个TfidfVectorizer实例并计算TF-IDF值
tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
feature_names = tfidf_vectorizer.get_feature_names_out()

# 创建一个词语索引到词向量的映射
word2vec_dict = {word: model.wv[word] for word in vocab}


# 计算每个文档的向量表示
def document_vector(doc):
    tfidf_vector = tfidf_vectorizer.transform([doc])
    tfidf_scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))

    doc_vector = np.zeros(model.vector_size)
    for word, score in tfidf_scores.items():
        if word in word2vec_dict:
            doc_vector += score * word2vec_dict[word]
    return doc_vector


# 获取每个文档的向量表示   每行代表一个文档
document_vectors = [document_vector(doc) for doc in corpus]

# 打印每个文档的向量表示
for i, vec in enumerate(document_vectors):
    print(f"Document {i + 1} vector: {vec}\n")
