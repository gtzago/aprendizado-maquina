# 9) Realize a classificacao da base de dados Car Evaluation (disponivel em
# http://archive.ics.uci.edu/ml/) usando o kNN. Realize 3-fold cross validation
# e, para cada rodada, use dois folds para a parte de calibracao e um fold para
# teste. A parte de calibracao o treinamento deve ser realizado usando um fold
# e a validacao do valor de k deve ser realizado usando o outro fold. A
# calibracao deve ser realizada de forma a maximizar a acuracia. Expresse os
# resultados em forma de acuracia media, macroprecision medio, macrorecall
# medio, tabela de contingencia medio (dado em porcentagem).

import classif_regres
import utils


x, y = utils.abrir_dados_car('./bases/car/car.data')

acc, c, pr = classif_regres.kfold_cross_validation(x, y, 3, utils.accuracy, classif_regres.knn, 3)

print 'acuracia media = '+ str(acc)
print 'Matriz de confusao media = '+ str(100*c)
print 'Macro Precisao media'+ str(pr[:,0])
print 'Macro Revocacao media'+ str(pr[:,1])