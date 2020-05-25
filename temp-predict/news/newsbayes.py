# -*- encoding: utf-8 -*-
"""
@File    : newsbayes.py
@Time    : 2020/5/25 15:51
@Author  : 王一宁
@Email   : wyn_365@163.com
"""
import pandas as pd
import jieba

# 数据 指定列名字
df_news = pd.read_table('D:\\APP\\PythonCharm2018\\workspace\\temp-predict\\data\\data.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()
df_news.tail()

df_news.shape

# 分词
content = df_news.content.values.tolist() #将每一篇文章转换成一个list
print (content[1000]) #随便选择其中一个看看

content_S = []
for line in content:
    current_segment = jieba.lcut(line) #对每一篇文章进行分词
    if len(current_segment) > 1 and current_segment != '\r\n': #换行符
        content_S.append(current_segment) #保存分词的结果

# 查看第1000个分词结果
content_S[1000]

# 专门展示分词后的结果
df_content=pd.DataFrame({'content_S':content_S})
df_content.head()

stopwords=pd.read_csv("D:\\APP\\PythonCharm2018\\workspace\\temp-predict\\data\\stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
stopwords.head(20)

# 筛选 停用词 没用的词
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word) # 保存
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content=pd.DataFrame({'contents_clean':contents_clean})
df_content.head()

df_all_words=pd.DataFrame({'all_words':all_words})
df_all_words.head()

print("=================================================")

# 词云展示
import numpy
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"],ascending=False)
words_count.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

wordcloud = WordCloud(font_path="",background_color="while",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()


# TF-IDF 提取关键词
import jieba.analyse #工具包
index = 2400 #随便找一篇文章就行
content_S_str = "".join(content_S[index]) #把分词的结果组合在一起，形成一个句子
print (content_S_str) #打印这个句子
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))#选出来5个核心词

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
df_train.tail()

# 制作数据集标签
df_train.label.unique()

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
df_train.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)

#x_train = x_train.flatten()
x_train[0][1]

words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index,word_index)
words[0]

print (len(words))

# 制作词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird'] #为了简单期间，这里4句话我们就当做4篇文章了
cv = CountVectorizer() #词频统计
cv_fit=cv.fit_transform(texts) #转换数据

print(cv.get_feature_names())
print(cv_fit.toarray())


print(cv_fit.toarray().sum(axis=0))

from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer(ngram_range=(1,4)) #设置ngram参数，让结果不光包含一个词，还有2个，3个的组合
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())


print(cv_fit.toarray().sum(axis=0))

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='word',lowercase = False)
feature = vec.fit_transform(words)
feature.shape

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
feature = vec.fit_transform(words)

feature.shape

# 磁带特征建模
from sklearn.naive_bayes import MultinomialNB #贝叶斯模型
classifier = MultinomialNB()
classifier.fit(feature, y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        #
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index,word_index)
test_words[0]

classifier.score(vec.transform(test_words), y_test)

# 制作TF-IDF特征
from sklearn.feature_extraction.text import TfidfVectorizer

X_test = ['卡尔 敌法师 蓝胖子 小小','卡尔 敌法师 蓝胖子 痛苦女王']

tfidf=TfidfVectorizer()
weight=tfidf.fit_transform(X_test).toarray()
word=tfidf.get_feature_names()
print (weight)
for i in range(len(weight)):
    print (u"第", i, u"篇文章的tf-idf权重特征")
    for j in range(len(word)):
        print (word[j], weight[i][j])

# 使用TF-IDF特征建模来观察结果
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

classifier.score(vectorizer.transform(test_words), y_test)

