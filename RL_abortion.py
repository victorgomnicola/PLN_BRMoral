#Manipulacao de dados
import numpy as np
import pandas as pd

##Validacao cruzada
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

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



nome_arquivo = "Dados/ep1_abortion_train.csv"
dados_aborto = pd.read_csv(nome_arquivo, delimiter=';')

X = dados_aborto.text
y = dados_aborto.abortion

folds = np.load("Dados/folds.npy", allow_pickle=True)

###Etapas do pré-processamento
n_grams = [1, 2, 3] #6
max_df = [0.5, 0.75]
min_df = [1, 2, 3, 4]
analyzer = 'words'
k_best_values = [200, 300, 500, 1000, 1200, 1500]

clf_c = [0.001, 0.005, 0.01, 0.05, .1, .5, 1, 5, 10, 50, 100]
solver = ['lbfgs']
clf_max_iter = [7000]

resultados = pd.DataFrame(columns=['Algoritmo', 'Ordem do teste', 'min_ngram', 'max_ngram', 'max_df', 'min_df',  'C', 'max_iter', 'kbest_value', 'avg_f1_macro', 'std_f1_macro'])
i = 0

####LACOS DO PRE-PROCESSAMENTO
for p1 in n_grams:
  for p2 in n_grams:
     if p1 <= p2:
       for max_d in max_df:
         for min_d in min_df:
           for k in k_best_values:
            ####LACOS DO CLASSIFICADOR
            for c in clf_c:
              for m in clf_max_iter:

                  #Aplica o pre-processamento
                  vectorizer = CountVectorizer(min_df = min_d, max_df = max_d, stop_words = stopwords.words('portuguese'), ngram_range = (p1, p2), lowercase = True)
                  X = vectorizer.fit_transform(dados_aborto.text).toarray()


                  #Seleção de features
                  if (k > X.shape[1]):
                    kvalue = 'all'
                  else:
                    kvalue = k
                  
                  sel = SelectKBest(k=kvalue)
                  best_sel = SelectKBest(k=kvalue)
                  best_fit = best_sel.fit(X, y)
                  X_train_best = best_fit.transform(X)

                  #Validacao_cruzada no classificador
                  classificador = LogisticRegression(random_state=0, max_iter = m)
                  scores = cross_validate(classificador, X_train_best, y, cv=folds, scoring=['f1_macro'])
                  print('TESTE ABORTO', i, ': f1_macro:', np.mean(scores['test_f1_macro']), 'accuracy std:', np.std(scores['test_f1_macro']))

                  #Preenche o data frame com os resultados
                  resultados.loc[i, 'Algoritmo'] = 'Regressão Logística'
                  resultados.loc[i, 'Ordem do teste'] = i
                  resultados.loc[i, 'min_ngram'] = p1
                  resultados.loc[i, 'max_ngram'] = p2
                  resultados.loc[i, 'max_df'] = max_d
                  resultados.loc[i, 'min_df'] = min_d
                  resultados.loc[i, 'C'] = c
                  resultados.loc[i, 'max_iter'] = m
                  resultados.loc[i,'kbest_value'] =  k
                  resultados.loc[i, 'avg_f1_macro'] = np.mean(scores['test_f1_macro'])
                  resultados.loc[i, 'std_f1_macro'] = np.std(scores['test_f1_macro'])


                  #Escrever os resultados na planilha final
                  arquivo_final = 'Resultados/abortion_resultados_regressao.csv'
                  resultados.to_csv(arquivo_final)

                  i += 1

