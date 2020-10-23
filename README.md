# PLN_BRMoral

O presente trabalho apresenta um estudo de diferentes classificadores aplicados a um
conjunto de dados formado por textos extra´ıdos do twitter que foram rotulados de acordo
com o posicionamento pol´ıtico dos autores [Santos and Paraboni 2019]. Dentre os t´opicos
dispon´ıveis, foram utilizados apenas os textos referentes `a aborto e controle de armas e
os posicionamentos poss´ıveis para cada texto s˜ao: a favor, contra ou neutro. Os textos
foram representados como bag of words formados por n-gramas de palavras com diversos
tamanhos e diferentes pr´e-processamentos aplicados, conforme detalhado na sec¸ ˜ao 2.
Para a classificac¸ ˜ao, foram selecionados quatro algoritmos: Multinomial Nayve Bayes,
Regress˜ao Log´ıstica, Support Vector Machine (SVM) e redes neurais multilayer perceptron
(MLP).

# Instruções
Foram desenvolvidos diferentes c´odigos, um para cada classificador e conjunto de dados,
totalizando 8 arquivos (e.g. RL abortion). Foi realizada uma padronizac¸ ˜ao nos
parˆametros de pr´e-processamento para reproduzir os resultados. Isso permite a paralelizac¸ ˜ao
dos treinos e testes, bem como a an´alise dos resultados. As bibliotecas necess´arias para
executar os c´odigos s˜ao: sklearn, pandas, numpy e nltk.
As execuc¸ ˜oes dos scripts devem ent˜ao ser feitas separadamente, cada qual com o
seu respectivo arquivo .py. Al´em disso, deve-se atentar para que a localizac¸ ˜ao relativa
dos scripts e todos os arquivos seja mantida (i.e. base de dados e resultados).

Assim, cada script ao ser executado extrai os dados da pasta Dados e, ao terminar
sua execuc¸ ˜ao, salva os resultados na pasta Resultados de acordo com uma nomenclatura
padr˜ao apresentada na ´arvore acima, mostrando a qual execuc¸ ˜ao o arquivo de resultados
se refere. Basta executar cada um da maneira preferida, seja via linha de comando
seja em um IDE.
