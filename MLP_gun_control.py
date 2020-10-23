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

folds = np.load("Dados/folds_armas.npy", allow_pickle=True)

n_grams = [1, 2, 3] #15
max_df = [0.5, 0.75] 
min_df = [1, 2, 3, 4]
analyzer = 'words'
kbest_values = [200, 300, 500, 1000, 1200, 1500]

hidden_layer_sizes = [(25, 25), (50, 50),  (100,100)]
activation = ['relu','tanh']
solver = ['adam']
alpha = [0.0001, 0.00001, 0.000001]
batch_size = [64]
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_init = [0.001, 0.0001, 0.00001]
early_stopping = [True]


resultados = pd.DataFrame(columns=['Algoritmo', 'Ordem do teste', 'min_ngram', 'max_ngram', 'max_df', 'min_df',  'max_features', 'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size', 'learning_rate', 'learning_rate_init', 'early_stopping','k_best_values', 'avg_f1_macro', 'std_f1_macro'])
i = 0

####LACOS DO PRE-PROCESSAMENTO
for p1 in n_grams:
    for p2 in n_grams:
        if p1 <= p2:
            for _max in max_df:
                for _min in min_df:
                    for k in kbest_values:
                        ####LACOS DO CLASSIFICADOR
                        for layer in hidden_layer_sizes:
                            for a in activation:
                                for s in solver:
                                    for alph in alpha:
                                        for batch in batch_size:
                                            for lr in learning_rate:
                                                for lrt in learning_rate_init:
                                                    for stop in early_stopping:

                                                        #Aplica o pre-processamento
                                                        vectorizer = CountVectorizer( min_df = _min, max_df = _max, stop_words = stopwords.words('portuguese'), ngram_range = (p1, p2), lowercase=True)
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
                                                        classificador = MLPClassifier(hidden_layer_sizes = layer, activation = a, solver = s, alpha = alph, batch_size = batch, learning_rate = lr, learning_rate_init = lrt, early_stopping = stop)

                                                        scores = cross_validate(classificador, X_train_best, y, cv=folds, scoring=['f1_macro'])
                                                        print('teste', i, ': f1_macro:', np.mean(scores['test_f1_macro']), 'accuracy std:', np.std(scores['test_f1_macro']))

                                                        #Preenche o data frame com os resultados
                                                        resultados.loc[i, 'Algoritmo'] = 'Multilayer Perceptron'
                                                        resultados.loc[i, 'Ordem do teste'] = i
                                                        resultados.loc[i, 'min_ngram'] = p1
                                                        resultados.loc[i, 'max_ngram'] = p2
                                                        resultados.loc[i, 'max_df'] = _max
                                                        resultados.loc[i, 'min_df'] = _min
                                                        resultados.loc[i, 'hidden_layer_sizes'] = layer
                                                        resultados.loc[i, 'activation'] = a
                                                        resultados.loc[i, 'solver'] = s
                                                        resultados.loc[i, 'alpha'] = alph
                                                        resultados.loc[i, 'batch_size'] = batch
                                                        resultados.loc[i, 'learning_rate'] = lr
                                                        resultados.loc[i, 'learning_rate_init'] = lrt
                                                        resultados.loc[i, 'early_stopping'] = 'True'
                                                        resultados.loc[i, 'k_best_values'] = min(k, X.shape[1])
                                                        resultados.loc[i, 'avg_f1_macro'] = np.mean(scores['test_f1_macro'])
                                                        resultados.loc[i, 'std_f1_macro'] = np.std(scores['test_f1_macro'])
                                                        

                                                        #Escrever os resultados na planilha final
                                                        arquivo_final = 'Resultados/gun_control_resultados_multilayer_perceptron.csv'
                                                        resultados.to_csv(arquivo_final)

                                                        i += 1

