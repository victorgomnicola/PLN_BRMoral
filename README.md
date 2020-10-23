# PLN_BRMoral

O presente trabalho apresenta um estudo de diferentes classificadores aplicados a um
conjunto de dados formado por textos extraídos do twitter que foram rotulados de acordo
com o posicionamento político dos autores [Santos and Paraboni 2019]. Dentre os tópicos
disponíveis, foram utilizados apenas os textos referentes à aborto e controle de armas e
os posicionamentos possíveis para cada texto são: a favor, contra ou neutro. Os textos
foram representados como bag of words formados por n-gramas de palavras com diversos
tamanhos e diferentes pré-processamentos aplicados, conforme detalhado na seção 2 do relatório preliminar.
Para a classificação, foram selecionados quatro algoritmos: Multinomial Nayve Bayes,
Regressão Logística, Support Vector Machine (SVM) e redes neurais multilayer perceptron
(MLP).

# Instruções
Foram desenvolvidos diferentes códigos, um para cada classificador e conjunto de dados,
totalizando 8 arquivos (e.g. RL abortion). Foi realizada uma padronização nos
parâmetros de pré-processamento para reproduzir os resultados. Isso permite a paralelização
dos treinos e testes, bem como a análise dos resultados. As bibliotecas necessárias para
executar os códigos são: sklearn, pandas, numpy e nltk.
As execuções dos scripts devem então ser feitas separadamente, cada qual com o
seu respectivo arquivo .py. Além disso, deve-se atentar para que a localização relativa
dos scripts e todos os arquivos seja mantida (i.e. base de dados e resultados).

Assim, cada script ao ser executado extrai os dados da pasta Dados e, ao terminar
sua execução, salva os resultados na pasta Resultados de acordo com uma nomenclatura
padrão. Basta executar cada um da maneira preferida, seja via linha de comando
seja em um IDE.
