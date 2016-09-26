__author__ = 'Sourabh'
import re
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
import Orange
from Orange import data
import pickle
import os
import glob
from classes import *


from classes.features2 import *

class NN11:
    logfile_uniquewords = {}
    uniquelogwords = {}
    uniquewords_map = {}
    map_index = 0
    lognum = 0
    featurelist = []
    headersWritten = False
    counts = []
    ssc_path = "G:/Users/skatti/Documents/logpredict/ssc_train_logs"
    sca_path = "G:/Users/skatti/Documents/logpredict/sca_train_logs"
    dsca_path = "G:/Users/skatti/Documents/logpredict/dsca_train_logs"
    testlogs_path = "G:/Users/skatti/Documents/logpredict/test_logs"
    filenames = [()]
    log_train_features = {}
    log_test_features = {}

    def setTestLogsDir(self, testlogsdir):
        self.testlogs_path=testlogsdir

    def openTrainModel(self, filetoread):
        data = Orange.data.Table(filetoread)
        return data

    def readLogfile (self, filetoread):
        with open(filetoread, 'r') as f:
            logmessages = f.readlines()
        f.close()
        self.lognum+=1
        return logmessages

    def printlog (self, logmessage):
        for line in logmessage:
            print (line)

    def printUniqueWords(self, choice):
        if choice == 1:
            print (self.uniquelogwords.__len__())
            for log, wordlist in self.logfile_uniquewords.iteritems():
                print ("Log#"),log
                for word, count in wordlist.items():
                    print (word, count)
        elif choice == 2:
            print (self.uniquelogwords.__len__())
            for word, count in self.uniquewords_map.iteritems():
                print (word, count)

        else:
            print ("Incorrect option")

    def vectorizeWords(self, logfile):
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\S\w+\b',min_df=1)
        x = vectorizer.fit_transform(logfile)
        self.counts = x.toarray()
        return vectorizer

    def getTF(self):
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(self.counts)
        return tfidf

    def setFeatures(self, vectorizer, log_class):
        tfidf = self.getTF()

        words = ('sca', 'debug', '-logfile', 'systemspec', 'args','server','scan', '.fpr', '.fmdalgeneralexception','operatingsystemmxbean', '.hibernate','.escalatinglog4jreceiver','show -runtime','warn','tomcat','.jdbc', 'master info', 'master fine', 'master warning', '.nst' )
        featurelist = []
        for word in words:
            index = vectorizer.vocabulary_.get(word)
            if index!=None:
                a = tfidf[:,index]
                featurelist.append(sum(a.data))
            else:
                featurelist.append(0.0)
        featurelistt = featurelist/np.amax(featurelist)

        featurelist.append(log_class)
        return featurelistt

    def extractFeatures(self):
        for log, log_keywords in self.logfile_uniquewords.iteritems():
            if (log_keywords.has_key("-b")):
                feature1=log_keywords["-b"]
            else: feature1=0

            if (log_keywords.has_key("Static")):
                feature2=log_keywords["Static"]
            else: feature2=0

            if (log_keywords.has_key("-logfile")):
                feature3=log_keywords["-logfile"]
            else: feature3=0

            if (log_keywords.has_key("com.fortify.SCAExecutablePath=")):
                feature4=log_keywords["com.fortify.SCAExecutablePath="]
            else: feature4=0

            if (log_keywords.has_key("[warn] com.fortify.systemspec - ========================== fortify context startup")):
                feature5=log_keywords["[warn] com.fortify.systemspec - ========================== fortify context startup"]
            else: feature5=0

            if (log_keywords.has_key("-debug")):
                feature6=log_keywords["-debug"]
            else: feature6=0

            if (log_keywords.has_key("   [debug] com.fortify.systemspec")):
                feature7=log_keywords["   [debug] com.fortify.systemspec"]
            else: feature7=0

            if (log_keywords.has_key("S/MIME")):
                feature8=log_keywords["S/MIME"]
            else: feature8=0

            if (log_keywords.has_key("server")):
                feature9=log_keywords["server"]
            else: feature9=0

            if (log_keywords.has_key("'path-type'")):
                feature10=log_keywords["'path-type'"]
            else: feature10=0

            if (log_keywords.has_key("[WARN]")):
                feature11=log_keywords["[WARN]"]
            else: feature11=0

            if (log_keywords.has_key("com.fortify.sca.Debug=")):
                feature12=log_keywords["com.fortify.sca.Debug="]
            else: feature12=0


            features = np.asarray([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12], dtype=float)
            self.addToFeatures(features)
            return features

    def addToFeatures(self, featurerow):
        #self.featurelist = [featurerow.__len__()]
        self.featurelist = np.vstack(featurerow)

    def writeHeaders(self, features, action):
        if action == 0:
            filename = 'traindata.tab'
        else:
            filename = 'testdata.tab'

    def writeFeature(self, featuress, logclass, action):
        #1 = SSC
        #2 = SCA
        #3 = SCA in debug mode
        rootdir = "G:/Users/skatti/Documents/logpredict/"

        if action == 0:
            filename = 'traindata.tab'
        else:
            filename = 'testdata.tab'


        column_names = ""
        variable_style = ""
        class_string = ""
        featurestring = ""
        a=1
        for feature in featuress:
            column_names = column_names + str(a) + '\t'
            variable_style = variable_style + 'd' + '\t'
            class_string = class_string + '\t'
            a+=1

        with open(filename, 'ab') as f:
            filetocheck = rootdir + filename
            if os.stat(filetocheck).st_size==0:
                f.write(column_names + 'logtype' + '\n')
                f.write(variable_style + 'd' + '\n')
                f.write(class_string + 'class' + '\n')
            for feature in featuress:
                    featurestring = featurestring + str(feature) + '\t'
            f.write(featurestring + str(logclass) + '\n')
        self.headersWritten=True



    def setupTrainingModel(self, file, logmessages, log_class):
        vectorizer = self.vectorizeWords(logmessages)
        featurelist = self.setFeatures(vectorizer, log_class)
        self.log_train_features[file]=featurelist

        return self.log_train_features

    def setupTestingModel(self, path):
        self.headersWritten = False

        #Setup testing data from testlogs_path folder
        with open(dir, 'r') as f:
            logmessages = f.readlines()
        f.close()
        try:
            vectorizer = self.vectorizeWords(logmessages)
            featurelist = self.setFeatures(vectorizer)
            self.log_test_features[testlog]=featurelist
            self.writeFeature(featurelist, 9, 1)
            print ".....Success"
        except:
            print ".....Failed"
        return self.log_test_features

    def logResults(self, logclass, probability):
        with open('trainresults.txt', 'ab') as f: f.write(logclass + " " + str(probability) + '\n')

    def setModel(self, traindata, testdata):
        neural = Orange.classification.neural.NeuralNetworkLearner(traindata, max_iter=50)
        logstoreturn = ([])
        target = 1
        print "\n\nProbabilities for test data:"

        for d, filename in zip(testdata, self.filenames):
            c = neural(d)
            if c == 0:
                logclass = "SSC"
                logstoreturn.append(filename)
            if c == 1: logclass = "SCA"
            if c == 2: logclass = "SCA in debug mode"


            ps = neural(d, Orange.classification.Classifier.GetProbabilities)
            print "There is a %s chance that this is a %s log file" % (ps[c], logclass)
            self.logResults(logclass, ps[c])

            #with open('trainresults.txt', 'ab') as f: f.write(logclass + " " + str(ps[c]) + '\n')
        neural_classifier = open('neural', 'wb')
        pickle.dump(neural, neural_classifier)
        neural_classifier.close()
        return logstoreturn

    def testModel(self, testdata):
        f = open('neural')
        neural_classifier = pickle.load(f)
        print ("\nTesting trained classifier...\n")
        for d in testdata:
            c = neural_classifier(d)
            if c == 0: logclass = "SSC"
            if c == 1: logclass = "SCA"
            if c == 2: logclass = "SCA in debug mode"
            ps = neural_classifier(d, Orange.classification.Classifier.GetProbabilities)
            print "There is a %s chance that this is a %s log file" % (ps[c], logclass)


#nn = NN()

#Specify how many times you want to train the neural network
i=1


#while i>=0:
    #Setup training and testing data sets
#    nn.trainModel()

    #Get training and testing data sets
#    train_data = nn.openTrainModel('traindata.tab')
#    test_data = nn.openTrainModel('testdata.tab')

    #Train the neural network then test it on the data
#    files = nn.setModel(train_data, test_data)
#    for filez in files:
#        print filez

    #Test data with the saved classifier, no training required
#    nn.testModel(test_data)
#    i-=1

#1 = SSC
#2 = SCA
#3 = SCA in debug mode








