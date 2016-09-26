__author__ = 'Sourabh'
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
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os
import glob
import operator
from classes import *
import re
import datetime
import calendar
import time


class wordstore:
    def __init__(self):
        self.wordprobability = {}
        self.counts = {}

    def addWord(self, wordprob):
        word = wordprob[0]
        prob = wordprob[1]
        if word in self.wordprobability.keys():
            sum_prob = float((self.wordprobability[word] * self.counts[word] + prob)) / float((self.counts[word] + 1))
            self.counts[word] += 1
            self.wordprobability[word] = sum_prob
        else:
            self.counts[word] = 1
            self.wordprobability[word] = prob
        return self.wordprobability[word]

    def printWords(self):
        for word, prob in self.wordprobability.items():
            print(word, prob)

    def resetWordStore(self):
        self.wordprobability.clear()
        self.counts.clear()


class ssc_logitem_nn:
    training_root = 'G:/Users/skatti/Documents/logpredict/li_train'
    counts = 0.0
    loglevels = {}
    featurelist = {}
    methodNames = {}
    stmethodNames = {}
    lemethodNames = {}
    messagecounts = 0.0
    charactercounts = 0.0
    shape = []

    def setupTrainingModel(self):
        files = os.listdir(self.training_root)
        for file in files:
            pathtofile = self.training_root + '/' + file

    def getShape(self, features1):
        lf = np.asarray(features1)
        return lf.shape

    def getWords(self, execution):
        logitems = []
        for logitem in execution.getlogitems():
            li_string = ""
            li_string = li_string + " " + logitem.logmessage + " " + logitem.methodname + " " + logitem.loglevel
            if logitem.hasExceptions():
                for exception in logitem.exception:
                    li_string = li_string + " " + exception.methodname + exception.description
                    if exception.hasStackTrace():
                        for st in exception.stacktrace:
                            li_string = li_string + " " + st.method + " " + st.filename
            logitems.append(li_string)
        return logitems

    def vectorizeWords(self, execution):
        vectorizer = CountVectorizer(ngram_range=(1, 3), lowercase=False, token_pattern=r'[\S+]+', min_df=1)
        x = vectorizer.fit_transform(execution)
        self.counts = x.toarray()
        return vectorizer

    def vectorizeLogItem(self, logmessage):
        vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r'[\S]+', min_df=1)
        x = vectorizer.fit_transform(logmessage)
        self.messagecounts = x.toarray()
        return vectorizer

    def vectorizeCharacters(self, logcharacters):
        vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r'\S+', min_df=1)
        x = vectorizer.fit_transform(logcharacters)
        self.charactercounts = x.toarray()
        return vectorizer

    def getTF(self):
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(self.counts)
        return tfidf

    def setFeatures(self, vectorizer, arrayToFill):
        tfidf = self.getTF()

    def setupTestingModel(self):
        files = os.listdir(self.training_root)
        for file in files:
            pathtofile = self.training_root + '/' + file
            print(file)

    def getLogLevelfeatures(self, loglevelprob):
        loglevel = loglevelprob[0]
        prob = loglevelprob[1]

        if loglevel in self.loglevels.keys():
            return self.loglevels[loglevel]
        else:
            count = self.loglevels.__len__() + 1
            self.loglevels[loglevel] = prob
            return count

    def getMethodNamefeatures(self, methodname):
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        if methodname in self.methodNames.keys():
            return self.methodNames[methodname]
        else:
            count = self.methodNames.__len__() + 1
            self.methodNames[methodname] = count
            return count

    def getSTmethodFeatures(self, methodname):
        if methodname in self.stmethodNames.keys():
            return self.stmethodNames[methodname]
        else:
            count = self.stmethodNames.__len__() + 1
            self.stmethodNames[methodname] = count
            return count

    def getLEfeatures(self, methodname):
        if methodname in self.lemethodNames.keys():
            return self.lemethodNames[methodname]
        else:
            count = self.lemethodNames.__len__() + 1
            self.lemethodNames[methodname] = count
            return count

    def getWordtf(self, li_vectorizer, li_tf, word):
        index = li_vectorizer.vocabulary_.get(word)
        if index is not None:
            a = li_tf[:, index]
            return [word, sum(a.data)]
        else:
            a = 0.0
            return [word, 0.0]

    def getMessageFeatures(self, logmessage):
        li_vectorizer = self.vectorizeLogItem(logmessage.split(" "))
        li_tfidf = self.getTF()

        wordscore = 0.0

        message_score = 0.0
        avg_message_score = 0.0

        critical_word_weight = 1500.0
        database_word_weight = 350.0
        ldap_word_weight = 20.0
        ssl_word_weight = 750.0

        critical_words = ['Error', 'unable', 'fail', 'exception', 'failed', 'invalid', 'error', 'Intercepted',
                          'nested exception', 'thrown']
        database_words = ['transaction', 'Hibernate', 'column', 'SQL', 'Communications link', 'artifact', 'SQLState']
        ldap_words = ['LDAP', 'com.fortify.manager.DAO.UsernameAndEmail', 'LDAP object']
        ssl_words = ['PKIX', 'valid certification', 'sun.security.provider.certpath.SunCertPathBuilderException:',
                     'simple bind', 'javax.net.ssl.SSLHandshakeException', ]

        search_words = (critical_words, database_words, ldap_words, ssl_words)
        word_weights = (critical_word_weight, database_word_weight, ldap_word_weight, ssl_word_weight)

        for category, weight in zip(search_words, word_weights):
            category_score = 0.0
            for word in category:
                wordscore = 0.0
                index = li_vectorizer.vocabulary_.get(word)
                if index != None:
                    b = li_tfidf[:, index]
                    a = sum(b.data)
                else:
                    a = 0.0
                # wordscore += (weight * float(a))
                category_score += (weight * float(a))
            message_score += category_score / float(category.__len__())

        return message_score

    def writeFeature(self, action, headers):
        if action == 0:
            filename = 'ssctrain.tab'
            arrayToFill = self.log_features
        else:
            filename = 'ssctest.tab'
            arrayToFill = self.testing_features

        rootdir = "G:/Users/skatti/Documents/logpredict/"
        column_names = ""
        variable_style = ""
        class_string = ""
        a = 1
        if arrayToFill.__len__() > 1:
            while a <= len(arrayToFill[0]):
                column_names = column_names + str(a) + '\t'
                variable_style = variable_style + 'd' + '\t'
                class_string = class_string + '\t'
                a += 1
        else:
            for featureslistp in arrayToFill:
                for feature in featureslistp:
                    column_names = column_names + str(a) + '\t'
                    variable_style = variable_style + 'd' + '\t'
                    class_string = class_string + '\t'
                    a += 1

        class_string = class_string[:-1]
        variable_style = variable_style[:-1]
        with open(filename, 'ab') as f:

            filetocheck = rootdir + filename
            if os.stat(filetocheck).st_size == 0:
                f.write(column_names + '\n')
                f.write(variable_style + 'd' + '\n')
                f.write(class_string + 'class' + '\n')
            for execution_features in arrayToFill:
                featurestring = ""
                for feature in execution_features:
                    featurestring = featurestring + str(feature) + '\t'
                f.write(featurestring + '\n')

    def createFeatureList(self, lihs1, lihs2, lihs3):
        target = 0
        linelist = []
        outputf = []
        i = 0
        features_to_return = []
        linum1 = []
        linum2 = []
        linum3 = []

        logrankings = (lihs1, lihs2, lihs3)

        # print(self.featurelist.__len__())

        try:
            for li in lihs1:
                linum1.append(li.linenumber)
        except:
            linum1 = -1

        try:
            for li in lihs2:
                linum2.append(li.linenumber)
        except:
            linum2 = -1

        try:
            for li in lihs3:
                linum3.append(li.linenumber)
        except:
            linum3 = -1

        for linenumber, featuress in self.featurelist.items():
            outputf = featuress.copy()
            target = 0

            try:
                if linenumber in linum1:
                    target = 1
            except:
                continue

            try:
                if linenumber in linum2:
                    target = 2
            except:
                continue

            try:
                if linenumber in linum3:
                    target = 3
            except:
                continue

            outputf.append(target)
            features_to_return.append(outputf)
            i += 1
        return features_to_return

    def printFeatures(self):
        for linenumber, featuress in self.featurelist.items():
            print(featuress)

    def getChartf(self, logitem):
        characters = (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', '.')
        logmessage_chartf = {}
        logmethod_chartf = {}
        causedBy_method_chartf = {}
        causedBy_description_chartf = {}

        li_vectorizer = self.vectorizeCharacters(logitem.methodname)
        li_tfidf = self.getTF()

        ## Get character tf from logitem METHOD
        for character in characters:
            index = li_vectorizer.vocabulary_.get(character)
            if index != None:
                try:
                    a = li_tfidf[:, index]
                    b = sum(a.data)
                except:
                    b = 0.0
            else:
                b = 0.0
            logmethod_chartf[character] = b

            ## Get character tf from logitem MESSSAGE
        li_vectorizer = self.vectorizeCharacters(logitem.logmessage.split(" "))
        li_tfidf = self.getTF()
        for character in characters:
            index = li_vectorizer.vocabulary_.get(character)
            if index != None:
                a = li_tfidf[:, index]
                b = sum(a.data)
            else:
                b = 0.0
            logmessage_chartf[character] = b
        for character in characters:
            logmessage_chartf[character] = 0.0

        ## Get character tf for last CAUSED BY exception in logitem
        if logitem.hasExceptions():
            exceptions = logitem.getExceptions()
            count = exceptions.__len__()
            le = exceptions[count - 1]

            li_vectorizer = self.vectorizeCharacters(le.methodname)
            li_tfidf = self.getTF()
            for character in characters:
                index = li_vectorizer.vocabulary_.get(character)
                if index != None:
                    a = li_tfidf[:, index]
                    b = sum(a.data)
                else:
                    b = 0.0
                causedBy_method_chartf[character] = b

            le_description = le.description
            li_vectorizer = self.vectorizeCharacters(le.description)

            li_tfidf = self.getTF()
            for character in characters:
                index = li_vectorizer.vocabulary_.get(character)
                if index != None:
                    a = li_tfidf[:, index]
                    b = sum(a.data)
                else:
                    b = 0.0
                causedBy_description_chartf[character] = b
        else:
            for character in characters:
                causedBy_description_chartf[character] = 0.0
                causedBy_method_chartf[character] = 0.0

        li_features = []
        limns = []
        lilms = []
        lemes = []
        ledes = []

        for limn, lilm, leme, lede in zip(logmethod_chartf.values(), logmessage_chartf.values(),
                                          causedBy_method_chartf.values(), causedBy_description_chartf.values()):
            limns.append(limn)
            lilms.append(lilm)
            lemes.append(leme)
            ledes.append(lede)

        li_features.extend(limns)
        li_features.extend(lilms)
        li_features.extend(lemes)
        li_features.extend(ledes)

        return li_features

    def writeFeatures(self, features, trainOnly):
        features1 = np.asarray(features)
        x, y = features1.shape

    def extractTrainingFeatures(self, parsedlog):
        self.featurelist.clear()

        ll = 'WARN||ERROR||Severe||Info'
        ospattern = 'OperatingSystemMXBean: (.*)'
        sscversionpattern = 'Version:([\d\.]+)'

        llc = re.compile(ll, re.IGNORECASE | re.DOTALL)
        osp = re.compile(ospattern, re.IGNORECASE | re.DOTALL)
        sscv = re.compile(sscversionpattern, re.IGNORECASE | re.DOTALL)

        score_sum = 0.0
        le_score_avg = 0.0
        previous_score = 0.0
        st_score_avg = 0.0
        change_sum = 0.0
        score_avg = 0.0
        change = 0.0
        change_avg = 0.0
        change_sum = 0.0
        scorechangerate = 0.0
        scorechangerate_avg = 0.0

        lews = wordstore()
        li_method_ws = wordstore()
        li_loglevel_ws = wordstore()
        stws = wordstore()
        lihs1 = []
        lihs2 = []
        lihs3 = []

        for execution in parsedlog.executions:
            high_score = 0.0
            li_words = self.getWords(execution)

            if li_words.__len__() > 0:
                li_vectorizer = self.vectorizeWords(li_words)
                li_tf = self.getTF()
                startline = execution.startline
                hs1 = 0
                hs2 = 0
                hs3 = 0

                lihs = []

                if execution.getlogitemlength() > 1000:
                    printStatusbar = True

                i = 1
                for logitem in execution.getlogitems():

                    if i == 1:
                        logitem_hs = logitem

                    lineFeature = []
                    le_message = ""

                    li_loglevel_prob = self.getWordtf(li_vectorizer, li_tf, logitem.loglevel)
                    feature1 = li_loglevel_ws.addWord(li_loglevel_prob)

                    try:
                        endline = logitem.linenumber
                        feature2 = float(int(endline) - int(startline)) / float(parsedlog.totallines)
                        startline = logitem.linenumber
                    except:
                        feature2 = 0.0

                    li_methodname_tf = self.getWordtf(li_vectorizer, li_tf, logitem.methodname)
                    feature3 = li_method_ws.addWord(li_methodname_tf)

                    try:
                        feature4 = float(logitem.linenumber) / float(parsedlog.getLineCount())
                    except:
                        feature4 = 0.0

                    feature5 = float(logitem.linenumber) / float(execution.endline)

                    character_features = np.asarray(self.getChartf(logitem))
                    for value in character_features:
                        lineFeature.append(value)

                    if logitem.hasExceptions():

                        st_score = 0.0
                        le_score = 0.0
                        logexec = logitem.getExceptions()
                        j = 0
                        for le in logexec:
                            j += 1
                            li_methodname_tf = self.getWordtf(li_vectorizer, li_tf, le.methodname)
                            le_score += lews.addWord(li_methodname_tf)
                            le_score_avg = float(le_score) / float(j)
                            le_message += le.description

                            if le.hasStackTrace():
                                st = le.stacktrace
                                p = 0
                                for stmn in st:
                                    p += 1
                                    st_methodname_tf = self.getWordtf(li_vectorizer, li_tf, stmn.method)
                                    st_score += stws.addWord(st_methodname_tf)
                                    st_score_avg = float(st_score) / float(p)

                    try:
                        le_message_score = self.getMessageFeatures(le_message)
                    except:
                        le_message_score = 0.0

                    try:
                        exception_score = float(st_score_avg) / float(le_score_avg)
                    except:
                        exception_score = 0.0

                    try:
                        message_score = self.getMessageFeatures(logitem.logmessage)
                    except:
                        message_score = 0.0

                    base_score = ((feature1) / (feature3) * (feature4) * (feature2))
                    score = base_score + exception_score + message_score + le_message_score

                    score_sum += score
                    score_avg = score_sum / float(i)
                    score_diff = score - score_avg

                    scorediff = score - previous_score

                    if not (previous_score == 0.0): scorechangerate = scorediff / float(previous_score)

                    if not (score_avg == 0.0): scorechangerate_avg = scorediff / score_avg

                    if not (score_avg == 0.0): change = (score - score_avg) / score_avg

                    change_sum += (change)
                    change_avg = float(change_sum) / float(i)

                    if scorechangerate == (np.nan or np.inf):
                        scorechangerate = 0.0
                        scorechangerate_avg = 0.0
                        scorediff = 0.0
                        change = 0.0
                        change_sum = 0.0
                        change_avg = 0.0

                    if score > hs3:
                        if score > hs2:
                            if score > hs1:
                                lihs1.append(logitem)
                                hs3 = hs2
                                hs2 = hs1
                                hs1 = score
                            else:
                                lihs2.append(logitem)
                                hs3 = hs2
                                hs2 = score
                        else:
                            lihs3.append(logitem)
                            hs3 = score
                    previous_score = score

                    lineFeature.extend(
                        [feature1, feature2, feature3, feature4, scorechangerate, scorechangerate_avg, change, change_avg,
                         feature5, score_avg, score])
                    self.featurelist[logitem.linenumber] = lineFeature
                    logitem_old = logitem

                    i += 1
            else:
                pass

        try:
            # print li_possibilities
            for li in lihs:
                print("Possible causes of error: %s on line %d" % (li.methodname, li.linenumber))
        except:
            print("Can't find the highest scoring logitem")
        featurez1 = self.createFeatureList(lihs1, lihs2, lihs3)
        featuress = np.asarray(featurez1).astype(float)
        featurez = featuress.tolist()

        return featurez
