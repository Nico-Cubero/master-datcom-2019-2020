#!/bin/bash

BASEDIR="/home/nico/moa-release-2019.05.0" #`dirname $0`/..
#BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=512m

java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" \
-javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"EvaluateInterleavedTestThenTrain -l (moa.classifiers.drift.SingleClassifierDrift -l"\
" bayes.NaiveBayes -d DDM) -s (ConceptDriftStream -s (generators.SEAGenerator -f 2) -d" \
 "(generators.SEAGenerator -f 3) -p 20000 -w 100) -i 100000"
