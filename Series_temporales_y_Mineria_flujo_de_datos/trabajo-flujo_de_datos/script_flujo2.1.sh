#!/bin/bash

if [ $# -ne 1 ]
	then echo "Especificar: $0 <Directorio base de MOA>"
		exit 1
fi

BASEDIR=$1
MEMORY=512m

seeds=(2 11 23)

for s in ${seeds[@]}
do
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l (LearnModel -l trees.HoeffdingTree"\
									"-s (generators.WaveformGenerator -i $s)"\
									"-m 1000000)"\
		"-s (generators.WaveformGenerator -i 4)"\
		"-i 1000000" > flujo2.1.1_seed$s.txt


	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
	-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluateInterleavedTestThenTrain -l (LearnModel -l trees.HoeffdingAdaptiveTree"\
									"-s (generators.WaveformGenerator -i $s)"\
									"-m 1000000)"\
		"-s (generators.WaveformGenerator -i 4)"\
		"-i 1000000" > flujo2.1.2_seed$s.txt

done
