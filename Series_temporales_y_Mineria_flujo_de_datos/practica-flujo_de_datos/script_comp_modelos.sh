#!/bin/bash

BASEDIR="/home/nico/moa-release-2019.05.0" #`dirname $0`/..
#BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=512m

seeds=(23 17 2246 76 9257 8 349 5 12165 51 \
		43 104 1073 7624 2 866 1355 1274 9543 4421 \
		5378 1007 1743 5888 1921 4211 3422 56 1020 370)

# Crear directorios si no existen
if [ ! -d "naive-bayes_results" ]
	then mkdir naive-bayes_results
fi

if [ ! -d "hoeffding-tree_results" ]
	then mkdir hoeffding-tree_results
fi

# Crear ficheros donde se almacenarán los resultados finales
echo -n "" > nb.txt
echo -n "" > ht.txt


for i in $(seq 0 29)
do
	# Ejecución con Naive Bayes
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l bayes.NaiveBayes " \
	"-s (generators.RandomTreeGenerator -i ${seeds[$i]} ) " \
		" -i 1000000 -f 10000" > "naive-bayes_results/nb${i}.txt"

	# Almacenar últimos valores de columna "Correct Classifier (Percent)"
	cat "naive-bayes_results/nb${i}.txt" | tail -n 1 | \
		sed -re 's/([^,]*,){4}([^,]*),.*/\2/' >> nb.txt

	# Ejecución con Hoeffding Trees
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l trees.HoeffdingTree " \
	"-s (generators.RandomTreeGenerator -i ${seeds[$i]} ) " \
		" -i 1000000 -f 10000" > "hoeffding-tree_results/ht${i}.txt"

	# Almacenar últimos valores de columna "Correct Classifier (Percent)"
	cat "hoeffding-tree_results/ht${i}.txt" | tail -n 1 | \
		sed -re 's/([^,]*,){4}([^,]*),.*/\2/' >> ht.txt
done
