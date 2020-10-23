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


nome_arquivo = "Dados/ep1_gun-control_train.csv"
dados_aborto = pd.read_csv(nome_arquivo, delimiter=';')


X = dados_aborto.text
y = dados_aborto['gun-control']

folds = np.load("Dados/folds.npy", allow_pickle=True)

n_grams = [1, 2, 3] #6
max_df = [0.5,0.75] 
min_df = [1, 2, 3, 4]
analyzer = 'words'
kbest_values = [200,300,500,1000,1200,1500]

clf_alpha = [0.001, 0.01, 0.1, 1.0]
clf_fit_prior = [1e-2, 1e-3, 1e-4, 1e-5]

resultados = pd.DataFrame(columns=['Algoritmo', 'Ordem do teste', 'min_ngram', 'max_ngram', 'max_df', 'min_df', 'alpha', 'fit_prior', 'kbest_value' ,'avg_f1_macro', 'std_f1_macro'])
i = 0
####LACOS DO PRE-PROCESSAMENTO
for p1 in n_grams:
    for p2 in n_grams:
        if p1 <= p2:
            for maxd in max_df:
                for mind in min_df:
                    for k in kbest_values:
                        ####LACOS DO CLASSIFICADOR
                        for alpha in clf_alpha:
                            for fit_prior in clf_fit_prior:

                                #Aplica o pre-processamento
                                vectorizer = CountVectorizer(min_df = mind, max_df = maxd, stop_words = stopwords.words('portuguese'), ngram_range = (p1, p2), lowercase=True)
                                X = vectorizer.fit_transform(dados_aborto.text).toarray()
                                    
                                #Validacao_cruzada no classificador
                                classificador = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

                                #Seleção de features
                                if (k > X.shape[1]):
                                    kvalue = 'all'
                                else:
                                    kvalue = k
                                    
                                sel = SelectKBest(k=kvalue)
                                best_sel = SelectKBest(k=kvalue)
                                best_fit = best_sel.fit(X, y)
                                X_train_best = best_fit.transform(X)

                                scores = cross_validate(classificador, X_train_best, y, cv=folds, scoring=['f1_macro'])
                                    
                                print('teste', i, ': f1_macro:', np.mean(scores['test_f1_macro']), 'accuracy std:', np.std(scores['test_f1_macro']))

                                #Preenche o data frame com os resultados
                                resultados.loc[i, 'Algoritmo'] = 'Multinomial Naive Bayes'
                                resultados.loc[i, 'Ordem do teste'] = i
                                resultados.loc[i, 'min_ngram'] = p1
                                resultados.loc[i, 'max_ngram'] = p2
                                resultados.loc[i, 'max_df'] = maxd
                                resultados.loc[i, 'min_df'] = mind
                                resultados.loc[i, 'alpha'] = alpha
                                resultados.loc[i, 'fit_prior'] = fit_prior
                                resultados.loc[i, 'kbest_value'] = min(k,X.shape[1])
                                resultados.loc[i, 'avg_f1_macro'] = np.mean(scores['test_f1_macro'])
                                resultados.loc[i, 'std_f1_macro'] = np.std(scores['test_f1_macro'])

                                #Escrever os resultados na planilha final
                                arquivo_final = 'Resultados/gun_control_resultados_multinomial_naive_bayes.csv'
                                resultados.to_csv(arquivo_final)

                                i += 1


