#!/bin/bash

if [ $# -ne 1 ]
	then echo "Especificar: $0 <Directorio base de MOA>"
		exit 1
fi

BASEDIR=$1
MEMORY=512m

seeds=(1 11 23)

for s in ${seeds[@]}
do
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l trees.HoeffdingTree"\
		"-s (generators.RandomRBFGeneratorDrift -s 0.001 -k 3 -a 7 -c 2 -n 3 -i $s -r $s)"\
		"-i 2000000 -f 100000" > flujo2.3.1_seed$s.txt


	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l trees.HoeffdingAdaptiveTree"\
		"-s (generators.RandomRBFGeneratorDrift -s 0.001 -k 3 -a 7 -c 2 -n 3 -i $s -r $s)"\
		"-i 2000000 -f 100000" > flujo2.3.2_seed$s.txt

done
