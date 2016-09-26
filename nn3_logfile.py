__author__ = 'Sourabh'

import re
from classes import Execution
from classes import Logitem
from classes import Logfile
from classes import LogException
from classes import Stacktrace
from classes import Parser
from scipy import *
from sklearn.neural_network import BernoulliRBM
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import *
from classes import features
from sklearn import linear_model, datasets, metrics
from sklearn.pipeline import Pipeline
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os
import glob
import operator
from classes import *
import os
import ssclog_learner


class parser:
    logfiles = {}
    words = ('sca', 'debug', '-logfile', 'systemspec', 'args', 'server', 'scan', '.fpr', '.fmdalgeneralexception',
             'operatingsystemmxbean', '.hibernate', '.escalatinglog4jreceiver', 'show -runtime',
             'com.fortify.manager.service.emailserviceimpl',
             'tomcat', 'ldap validation', 'master info', 'master fine', 'master warning', '.nst')

    def __init__(self):
        print("Parsing logfile features")



    def convertToWords(self, ssclog):
        log_as_string = []
        for executionn in ssclog.executions:
            execution_as_string = []
            if executionn.hasProperties():
                for property in executionn.properties:
                    execution_as_string.append(property)
            for logitemm in executionn.getlogitems():
                logstring = logitemm.methodname + " " + logitemm.loglevel + " " + logitemm.logmessage
                if logitemm.hasExceptions():
                    for exception in logitemm.exception:
                        logstring = logstring + " " + exception.methodname + exception.description

                        if exception.hasStackTrace():
                            for st in exception.stacktrace:
                                logstring = logstring + " " + st.method
                execution_as_string.append(logstring)
            log_as_string.append(execution_as_string)
        return log_as_string

    def vectorizeWords(self, logfile):
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\S+', min_df=1)
        x = vectorizer.fit_transform(logfile)
        self.counts = x.toarray()
        return vectorizer

    def getTF(self):
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(self.counts)
        return tfidf

    def readFile(self, dir):
        with open(dir, 'r') as f:
            lflines = f.readlines()
        f.close()
        return lflines

    def parse(self, parsedlog, product):
        featurelist = []
        logfilestring = self.readFile(parsedlog)
        loglinedefaultpattern = '([0-9\-]+\s[0-9\:\,]+)\s+\[(\w+)\]\s([\.\w]+)\s\-\s([a-z0-9\s\W\d]+)+'
        sscconfigpattern = '([A-Z]+)\s[\d\-\s\,\:]+\[([\w\.]+)\][\-\s]+(.*)'
        rg = re.compile(loglinedefaultpattern, re.IGNORECASE | re.DOTALL)
        rc = re.compile(sscconfigpattern, re.IGNORECASE | re.DOTALL)
        ssc_count = 0
        config_count = 0
        try:
            vectorizer = self.vectorizeWords(logfilestring)
            # print vectorizer.get_feature_names()
            tfidf = self.getTF()
            try:
                for word in self.words:
                    index = vectorizer.vocabulary_.get(word)
                    if index is not None:
                        a = tfidf[:, index]
                        featurelist.append(sum(a.data))
                    else:
                        featurelist.append(0.0)

            except:

                featurelist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0]
        except:
            featurelist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0]

        for line in logfilestring:
            if rg.match(line):
                ssc_count += 1
            if rc.match(line):
                config_count += 1
        featurelist.append(ssc_count)
        featurelist.append(config_count)

        output = 0
        if product == 'ssc': output = 1
        if product == 'sca': output = 2
        if product == 'dsca': output = 3

        featurelist.append(output)
        return featurelist
