from scipy import *
import re
from time import time
import numpy as np
from storage import storage
from gensim import corpora
from gensim.models import Word2Vec, Doc2Vec, Phrases
import os
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import pickle


class db_trainer:
    dbwriter = storage.db_store()

    def __init__(self):
        # nltk.download()
        pass

    def getlogs(self):
        selectst = "SELECT * from logfiles"

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor()
        cursor.execute(selectst)

        logfiles = cursor.fetchall()

        self.dbwriter.closeconnection()

        return logfiles

    def getexecutions(self, lf_id):
        selectst = ("SELECT * from executions where logfile_id=%d" % lf_id)

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor()
        cursor.execute(selectst)

        executions = cursor.fetchall()

        self.dbwriter.closeconnection()
        return executions

    def getlogitems(self, exec_id):
        selectst = ("SELECT * from logitems where execution_id=%d" % exec_id)

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor()
        cursor.execute(selectst)

        logitems = cursor.fetchall()

        self.dbwriter.closeconnection()
        return logitems

    def getseedkeywords(self):
        selectst = "SELECT * from seed_words"

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor()
        cursor.execute(selectst)

        lf_keywords = cursor.fetchall()

        self.dbwriter.closeconnection()
        return np.asarray(lf_keywords)

    def sanitizelogfile(self, logfile):
        sanitized_lf = []  # Used to store the sanitized logfile, stripped of special characters and spaces
        continuous_lf = []
        tm = Word2Vec(iter=1,
                      min_count=1)  # Used to create the Word2Vec model. Contains a list of logitem as strings, rather than tuples
        for execution in logfile:
            sanitized_exec = []
            continuous_exec = []
            for logitem_db in execution[0]:
                li_raw = str(logitem_db[1]) + " " + str(logitem_db[2]) + " " + str(logitem_db[3])
                liiraw = ''
                li_temp = re.split('[\s\=\:\.\\\[\]\(\)\-\_\/\\\\\'\;\{\}\+\=\<\>\@\,\*\"\^\%\$\#\!\&\~\`\|\?]+',
                                   li_raw)
                sanitized_li = [x for x in li_temp if x]
                sanitized_num_li = [x for x in sanitized_li if re.match(r'\D+', x)]

                # for term in sanitized_li:
                #     liiraw = liiraw + " " + term
                sanitized_exec.append(sanitized_num_li)
                continuous_lf.append(sanitized_num_li)
            sanitized_lf.append(sanitized_exec)
            # continuous_lf.append(continuous_exec)
        tm.build_vocab(continuous_lf)
        return sanitized_lf, tm, continuous_lf

    def sanitizetestlogfile(self, logfile):
        sanitized_lf = []  # Used to store the sanitized logfile, stripped of special characters and spaces
        continuous_lf = []
        tm = Word2Vec(iter=1, min_count=1)  # Used to create the Word2Vec model. Contains a list of logitem as strings, rather than tuples
        for execution in logfile.getExecutions():
            sanitized_exec = []
            continuous_exec = []
            for logitem in execution.getlogitems():
                li_raw = str(logitem.methodname) + " " + str(logitem.logmessage) + " " + str(logitem.logmessage)
                liiraw = ''
                li_temp = re.split('[\s\=\:\.\\\[\]\(\)\-\_\/\\\\\'\;\{\}\+\=\<\>\@\,\*\"\^\%\$\#\!\&\~\`\|\?]+',
                                   li_raw)
                sanitized_li = [x for x in li_temp if x]
                sanitized_num_li = [x for x in sanitized_li if re.match(r'\D+', x)]

                # for term in sanitized_li:
                #     liiraw = liiraw + " " + term
                sanitized_exec.append(sanitized_num_li)
                continuous_lf.append(sanitized_num_li)
            sanitized_lf.append(sanitized_exec)
            # continuous_lf.append(continuous_exec)
        tm.build_vocab(continuous_lf)
        return sanitized_lf, tm, continuous_lf

    def addLFwordstodictionary(self, logfile):

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor()

        for executt in logfile:
            for logitem in executt:
                for term in logitem:
                    try:
                        cursor.execute("Select * from dictionary_words where keyword='%s'" % term)
                        results = cursor.fetchall()
                        if not results:
                            cursor.execute('INSERT into dictionary_words VALUES(%s)', term)
                    except Exception as e:
                        pass

        self.dbwriter.connection.commit()

        self.dbwriter.closeconnection()

    def initializemodel(self, w2v_models):
        print("\n======== Initiliazing models ========")
        print("\nThere are %d models" % w2v_models.__len__())
        for i in w2v_models.keys():
            model = w2v_models[i]
            print("\nModel #%d" % i)
            print("\tThere are %d words in this model" % model.vocab.__len__())
            for q in range(0, 20):
                print(model.index2word[q])

    def savemodel(self, model):
        currentfolder = os.getcwd()
        savefolder = currentfolder + '//models'
        numfiles = os.listdir(savefolder).__len__() + 1
        finalsavepath = savefolder + "//my-" + str(numfiles) + ".model"
        model.save(finalsavepath)

    def loadmodel(self):
        currentfolder = os.getcwd()
        loadfolder = currentfolder + '//models//my.model'
        loaded_model = Word2Vec.load(loadfolder)
        return loaded_model

    def updatedictionarytf(self):
        selectst_tf = "SELECT * from dictionary_words"
        sumst = "SELECT sum(count) from dictionary_words"
        print("Updating term frequency for dictionary words")

        self.dbwriter.startconnection()
        cursortf = self.dbwriter.getcursor()
        cursortf.execute(selectst_tf)
        dict_raw = cursortf.fetchall()

        cursortf.execute(sumst)
        total_count = cursortf.fetchone()

        for word in dict_raw:
            word_tf = float(word[2]) / float(total_count[0])
            updatest_tf = "UPDATE dictionary_words set term_frequency=%f where keyword='%s'" % (word_tf, word[1])
            cursortf.execute(updatest_tf)
            self.dbwriter.connection.commit()
        self.dbwriter.closeconnection()

    def generatedictionary(self, loglimit=None):
        lflist = self.getlogs()
        print("There are %d logfiles in the database" % lflist.__len__())

        limasterlist = []

        i = 0
        if not loglimit:
            loglimit = lflist.__len__()

        all_lf = []
        w2v_dictionary = {}

        while i < loglimit:
            lf = lflist[i]
            lf_id = lf[0]
            print("\tRunning for log# %d" % lf_id)
            lft = []
            executionlist = self.getexecutions(lf_id)
            for ex in executionlist:
                execa = []
                exec_id = ex[0]
                logitemlist = self.getlogitems(exec_id)
                execa.append(logitemlist)
                lft.append(execa)

            all_lf.append(lft)
            stlf, w2v_model, ctlf = self.sanitizelogfile(lft)

            self.addLFwordstodictionary(stlf)
            w2v_dictionary[lf_id] = w2v_model
            for line in ctlf:
                limasterlist.append(line)
            i += 1
        self.featurizeLF_db(w2v_dictionary)
        # self.initializemodel(w2v_dictionary)

    def getfeatures(self, logfile):
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
                        [feature1, feature2, feature3, feature4, scorechangerate, scorechangerate_avg, change,
                         change_avg,
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

    def initializedictionary(self):
        selectst = 'SELECT keyword from dictionary_words'
        dictwords = []
        self.dbwriter.startconnection()
        dcursor = self.dbwriter.getcursor()
        dcursor.execute(selectst)
        dictwords_db = dcursor.fetchall()
        self.dbwriter.closeconnection()
        wrodz = ''
        for word in dictwords_db:
            print(word[0])
            wrodz += " " + word[0]
            dictwords.append(word[0])

        model = Doc2Vec(dictwords)
        return model

    ## Generate features for each log file in the database. Take the index of the 20 most frequent words.
    def featurizeLF_db(self, w2v_dictionary):
        features_to_write = 40

        print(w2v_dictionary.keys().__len__())
        print("\n======== Initiliazing models ========")
        print("\nThere are %d models" % w2v_dictionary.__len__())
        for i in w2v_dictionary.keys():
            model = w2v_dictionary[i]
            print("\nModel #%d" % i)
            print("\tThere are %d words in this model" % model.vocab.__len__())
            most_frequent_words = model.index2word[0:features_to_write]
            # print(most_frequent_words)
            kwids = self.dbwriter.getdictionarykeywordsbyid(most_frequent_words)
            self.dbwriter.writeLFfeatures(kwids, i)
            # print(kwids)

    def trainLFfeatures(self):
        # Use a Naive Bayes classifier
        lf_nb_predictor = GaussianNB()

        lf_nn = MLPClassifier(hidden_layer_sizes=(1000, 100), activation='relu', solver='adam', batch_size=3,
                              learning_rate='invscaling', learning_rate_init=1.0, verbose=True)

        print("\n======== Starting LogFile classification training ========")
        features, targetoutputs = self.dbwriter.getLFfeatures()

        lf_nb_predictor.fit(features, targetoutputs)
        lf_nn.fit(features, targetoutputs)

        # Save the trained classifier for future predictions
        path_to_save = 'C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/nb_models/nb1'
        path_to_save_nn = 'C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/nb_models/nn'
        self.saveclassifier(path_to_save, lf_nb_predictor)
        self.saveclassifier(path_to_save_nn, lf_nn)

        return lf_nb_predictor

    def saveclassifier(self, path, classifier):
        with open(path, 'wb') as pf:
            pickle.dump(classifier, pf)

    def loadclassifier(self, path):
        with open(path, 'rb') as pf:
            cls = pickle.load(pf)
        return cls

    def testLFfeatures(self, classifier):
        features, targetoutputs = self.dbwriter.getLFfeatures()
        predictions = np.asarray(classifier.predict(features)).astype(int)
        print("\n************Naive Bayes RESULTS*************")
        for to, tp in zip(targetoutputs, predictions):
            value = 'false'
            if to == tp:
                value = ''
            print(to, tp, '%s' % value)

        path_to_load = 'C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/nb_models/nn'
        cls = self.loadclassifier(path_to_load)
        predictions_clf = np.asarray(cls.predict(features)).astype(int)
        print("\n************Testing with pickled model MLP RESULTS*************")
        for to, tp in zip(targetoutputs, predictions_clf):
            value = 'false'
            if to == tp:
                value = ''
            print(to, tp, '%s' % value)

    def test(self):
        self.dbwriter.test()

    def setupLFtestdictionar(self, parsedlog):
        num_features = 20
        stf, tm, clf = self.sanitizetestlogfile(parsedlog)
        most_frequent_words = tm.index2word[0:num_features]

        kwids = self.dbwriter.getdictionarykeywordsbyid(most_frequent_words)
        mfw = kwids.__len__()
        while mfw < num_features:
            kwids.append(-1)
            mfw += 1

        return kwids


# dbt = db_trainer()
# mod = dbt.loadmodel()
# print(mod.most_similar(positive=["Unable", "to", "start", "job", "scheduler"], negative=["INFO"]))
# dbt.generatedictionary()
# dbt.updatedictionarytf()
# dictionary = dbt.initializedictionary()
