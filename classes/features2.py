__author__ = 'Sourabh'

import re
from datetime import datetime
import hashlib
from classes import Logfile
from classes import *
from classes import Logitem
from classes.Logfile import *
from classes import Execution
from classes.Execution import *
from classes.features import *
from classes.features2 import *
from classes.LogException import *
from classes.Stacktrace import *
import re
import string

from classes import LogException, Execution, Stacktrace, Logitem
uniquewords = {}

class Features2:
    uniquewords = {}


    def extractUniqueWords(self, ssclog):
        for executionn in ssclog.executions:
            #assert isinstance(executionn, Execution)
            execution_words = {}

            execution_as_string = ""
            for logitemm in executionn.getlogitems():
                assert isinstance(logitemm, Logitem.Logitem)
                logstring = logitemm.methodname + " " + logitemm.loglevel + " " + logitemm.logmessage
                if logitemm.hasExceptions():
                    for exception in logitemm.exception:
                        assert isinstance(exception, Logexception)
                        logstring = logstring + " " + exception.methodname + exception.description

                        if exception.hasStackTrace():
                            for st in exception.stacktrace:
                                assert isinstance(st, Stacktrace)
                                logstring = logstring + " " + st.method
                execution_as_string = execution_as_string + " " + logstring
                wordslist = logstring.split()
                for word in wordslist:
                    if word in execution_words.keys():
                        execution_words[word] = execution_words[word] + 1
                    else:
                        execution_words[word]=1
            self.uniquewords[executionn]=execution_words
        return self.uniquewords

    def normalizeWords(self):
        for execution, keywords in self.uniquewords.items():
            total = float(sum(keywords.values()))
            for keyword in keywords:
                self.uniquewords[execution][keyword]= float(self.uniquewords[execution][keyword])/total

    def keywordProb(self, execution, keyword):
        if not (execution in self.uniquewords.keys() and keyword in self.uniquewords[execution].keys()): return 0.0
        else:
            test = (self.uniquewords[execution][keyword])
            return self.uniquewords[execution][keyword]

    def executionProb(self, execution):
        if not (execution in self.uniquewords.keys()): return 0.0
        count = self.uniquewords[execution].__len__()
        total = 0.0
        for executions, keywords in self.uniquewords.items():
            total = total + keywords.__len__()
        return float(count)/total

    def weightedProbabability(self, keyword, execution):
        weight=5.0
        ap=1.0
        # Calculate current probability
        basicprob = self.keywordProb(execution, keyword)

        # Count the number of times this feature has appeared in
        # all categories
        totals = 0.0
        for execution in self.uniquewords.keys():
            totals += float(self.keywordProb(execution, keyword))

        # Calculate the weighted average
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

    def docprob(self, execution):
        p = 1
        for keyword in self.uniquewords[execution].keys():
            p *= self.weightedProbabability(keyword, execution)
        return p

    def prob(self, keyword, execution):
        execution_prob = {}
        if not (execution in self.uniquewords.keys() or keyword in self.uniquewords[execution].keys()): return 0.0

        for execution, keywords in self.uniquewords.items():
            execprob = self.executionProb(execution)
            docprob = float(self.docprob(execution))
            execution_prob[execution] = docprob * execprob
        return execution_prob

    def executionSimilarity(self, execution):
        execution_probability = {}

        #loop over all keywords in the execution
        for exec_compare in self.uniquewords.keys():
            keywordprob = 0.0
            for keyword, probability in self.uniquewords[execution].items():
                if not (keyword in self.uniquewords[exec_compare].keys()): keywordprob = 0.0
                else:
                    keywordprob += float(probability*self.uniquewords[exec_compare][keyword])
            execution_probability[exec_compare]=keywordprob
        return execution_probability











