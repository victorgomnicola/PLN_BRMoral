#Manipulacao  de dados
import numpy as np
import pandas as pd

##Validacao
from sklearn.metrics import f1_score


#classificador
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer

##pre-preocessamento
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_selection import SelectKBest, chi2


nome_arquivo = "Dados/ep1_gun-control_train.csv"
dados_treino = pd.read_csv(nome_arquivo, delimiter=';')

nome_arquivo = "Dados/ep1_gun-control_test.csv"
dados_teste = pd.read_csv(nome_arquivo, delimiter=';')

X_treino = dados_treino.text
y_treino = dados_treino['gun-control']


X_teste = dados_teste.text


folds = np.load("Dados/folds.npy", allow_pickle=True)

n_grams = 1 #6
min_df = 1
max_df = 0.75
analyzer = 'words'
kbest_values = 1000

clf_alpha = 0.001
clf_fit_prior = 1e-3

#Aplica o pre-processamento
vectorizer = CountVectorizer(min_df = min_df, max_df = max_df, stop_words = stopwords.words('portuguese'), ngram_range = (n_grams, n_grams), lowercase = True, analyzer = 'word')
X_treino = vectorizer.fit_transform(dados_treino.text).toarray()

vectorizer = CountVectorizer(min_df = min_df, max_df = max_df, stop_words = stopwords.words('portuguese'), ngram_range = (n_grams, n_grams), lowercase = True, analyzer = 'word', max_features = 1000)
X_teste = vectorizer.fit_transform(dados_teste.text).toarray()

best_sel = SelectKBest(k = kbest_values)
best_fit = best_sel.fit(X_treino, y_treino)
X_train_best = best_fit.transform(X_treino)


print(X_train_best.shape)
#Validacao_cruzada no classificador
classificador = MultinomialNB(alpha = clf_alpha, fit_prior = clf_fit_prior)

modelo = classificador.fit(X_train_best, y_treino)

y_pred = modelo.predict(X_teste)

for i in range(1, X_teste.shape[0]):
    print(i, ':', y_pred[i])

