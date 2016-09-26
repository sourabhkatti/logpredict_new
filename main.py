__author__ = 'Sourabh'
from classes import Logfile
from classes import *
from classes import Logitem
from classes.Logfile import *
from classes import Execution
from classes.Execution import *
from classes.features import *
from classes.LogException import *
from classes.Stacktrace import *
import re
import string


#storage.createTables()


file3 = 'ssc-t.log.1'
file1 = 'Tomcat_ssc.log'
file2 = 'singleexception.log'
filetoread1 = 'sscjdbc.log'
#files = (file1, file2, file3)
files = (file2,)
ssclogclass = Logfile()
i=0
while i<1:
    for filetoread in files:
        ssclog = ssclogclass.parseFile(filetoread)

        logfileWeights = {}

        assert isinstance(ssclog, Logfile)

        features = Features()
        #logwords = features.tab.convertToWords(ssclog)
        #uniquewords = features.tab.getwords(logwords)

        logfileWeights= features.extractFeatures(ssclog)

        #print (features.tab.keywordCount('FATAL', 'train'))
#        features.tab.printFeatures()

        #storage.writeLog(ssclog)
        #storage.writeExecutionFeatures(logfileWeights)


        classifier = features.getClassifier()

#        classifier.printRankedFeatures()

        #print (test1)

        nb = naivebayes()
        nb.setClassifier(classifier)
        nb.setExecutionFeature(logfileWeights)


        logfileWeights= features.extractFeatures(ssclog)
        features.extractFeatures(ssclog)

        execution_probability = features.getExecProb()
        logitem_probability = features.classify.rankKeywords()
        execution_probability2 = features.classify.rankCategories()

        features.extractFeatures(ssclog)

        print ("Most frequent category: ")
        print (execution_probability2)
        print ("\n")

        cor = nb.prob('report', 'port_issue')
        for execution, probability in cor.items():
            print (execution.startline, probability)


    i=i+1