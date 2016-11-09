from scipy import *
import re
from time import time
import numpy as np
from storage import storage
from gensim.models import Word2Vec, Doc2Vec, Phrases
import os
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from sknn.mlp import Classifier, Layer
from sklearn.metrics import accuracy_score, precision_score


class db_trainer:
    dbwriter = storage.db_store()

    def __init__(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.abspath(os.curdir)

    def getlogs(self, startfromlast=True):

        if startfromlast:
            try:
                # Get the id of the last logfile feature to be written
                lastid = "SELECT max(log_id) from lf_features"
                self.dbwriter.startconnection()
                cursor = self.dbwriter.getcursor()
                cursor.execute(lastid)
                lastlogfile = np.squeeze(np.asarray(cursor.fetchone()).astype(int))

                # Get all logfiles greater than the id
                selectst = ("SELECT * from logfiles where id>%d" % lastlogfile)
                cursor.execute(selectst)
                logfiles = cursor.fetchall()
                return logfiles

            except:
                selectst = "SELECT * from logfiles"

                self.dbwriter.startconnection()
                cursor = self.dbwriter.getcursor()
                cursor.execute(selectst)

                logfiles = cursor.fetchall()

                self.dbwriter.closeconnection()

        else:
            selectst = "SELECT * from logfiles"

            self.dbwriter.startconnection()
            cursor = self.dbwriter.getcursor()
            cursor.execute(selectst)

            logfiles = cursor.fetchall()

            self.dbwriter.closeconnection()

        return logfiles

    def getLFfeatures(self):
        selectst = "SELECT * from lf_features"

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

    def sanitizelogfile_db(self, logfile):
        sanitized_lf = []  # Used to store the sanitized logfile, stripped of special characters and spaces
        continuous_lf = []

        # Used to create the Word2Vec model. Contains a list of logitem as strings, rather than tuples
        tm = Word2Vec(iter=1, min_count=1, size=20)

        for execution in logfile:
            sanitized_exec = []
            for logitem_db in execution[0]:
                li_raw = str(logitem_db[1]) + " " + str(logitem_db[2]) + " " + str(logitem_db[3])
                li_temp = re.split('[\s\=\:\.\\\[\]\(\)\-\_\/\\\\\'\;\{\}\+\=\<\>\@\,\*\"\^\%\$\#\!\&\~\`\|\?]+',
                                   li_raw)
                sanitized_li = [x for x in li_temp if (x and re.match(r'\D+', x))]
                # sanitized_num_li = [x for x in sanitized_li if re.match(r'\D+', x)]

                sanitized_exec.append(sanitized_li)
                continuous_lf.append(sanitized_li)
            sanitized_lf.append(sanitized_exec)
        tm.build_vocab(continuous_lf)
        return sanitized_lf, tm, continuous_lf

    def sanitizelogfile_object(self, logfile):
        sanitized_lf = []  # Used to store the sanitized logfile, stripped of special characters and spaces
        continuous_lf = []

        # Used to create the Word2Vec model. Contains a list of logitem as strings, rather than tuples
        tm = Word2Vec(iter=1, min_count=1, size=20)

        for execution in logfile.getExecutions():
            sanitized_exec = []
            for logitem in execution.getlogitems():
                li_raw = str(logitem.methodname) + " " + str(logitem.logmessage) + " " + str(logitem.logmessage)
                li_temp = re.split('[\s\=\:\.\\\[\]\(\)\-\_\/\\\\\'\;\{\}\+\=\<\>\@\,\*\"\^\%\$\#\!\&\~\`\|\?]+',
                                   li_raw)
                sanitized_li = [x for x in li_temp if (x or re.match(r'\D+', x))]
                # sanitized_num_li = [x for x in sanitized_li if re.match(r'\D+', x)]

                sanitized_exec.append(sanitized_li)
                continuous_lf.append(sanitized_li)
            sanitized_lf.append(sanitized_exec)
        tm.build_vocab(continuous_lf)
        return sanitized_lf, tm, continuous_lf

    def addLFwordstodictionary(self, w2v_model):

        self.dbwriter.startconnection()
        cursor = self.dbwriter.getcursor(asdict=True)

        for term, values in w2v_model.vocab.items():
            count = values.count
            try:
                cursor.execute("Select * from dictionary_words where keyword='%s'" % term)
                results = cursor.fetchall()
                if not results:
                    cursor.execute('INSERT into dictionary_words VALUES(%s, %d)', (term, count))
                else:
                    count += results[0]['count']
                    cursor.execute('UPDATE dictionary_words set count=%d where keyword=%s', (count, term))
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

    def count_to_tf(self, counts):
        sumst = "SELECT sum(count) from dictionary_words"

        self.dbwriter.startconnection()
        cursortf = self.dbwriter.getcursor()

        cursortf.execute(sumst)
        total_count = cursortf.fetchone()

        return np.divide(counts, total_count)

    def generatedictionary(self, queue, loglimit=None, featurize=True):
        lflist = self.getlogs()
        print("There are %d logfiles in the database" % lflist.__len__())

        limasterlist = []

        i = 0
        if not loglimit:
            loglimit = lflist.__len__()

        all_lf = []
        while i < loglimit:
            try:
                if queue.queue[-1].find('trainstop') is not -1:
                    queue.put("trainstop - logfile loop")
                    break
            except Exception as e:
                print("breaking", e)
                pass
            lf = lflist[i]
            lf_id = lf[0]
            print("\tRunning for log# %d" % lf_id)
            lft = []
            executionlist = self.getexecutions(lf_id)
            if executionlist.__len__() > 1:
                for ex in executionlist:
                    try:
                        if queue.queue[-1].find('trainstop') is not -1:
                            queue.put("trainstop - execution loop")
                            break
                    except Exception as e:
                        pass
                    execa = []
                    exec_id = ex[0]
                    logitemlist = self.getlogitems(exec_id)
                    execa.append(logitemlist)
                    lft.append(execa)
            else:
                try:
                    exec_id = executionlist[0][0]
                    execa = []
                    logitemlist = self.getlogitems(exec_id)
                    execa.append(logitemlist)
                    lft.append(execa)
                except:
                    i += 1
                    continue

            all_lf.append(lft)
            stlf, w2v_model, ctlf = self.sanitizelogfile_db(lft)

            # Write all unique words to dictionary_words table
            self.addLFwordstodictionary(w2v_model)
            for line in ctlf:
                limasterlist.append(line)
            i += 1

            if featurize:
                self.featurizeLF_db(w2v_model, lf_id)

                # self.initializemodel(w2v_dictionary)

    def getfeatures(self, parsedlog):
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

        lihs1 = []
        lihs2 = []
        lihs3 = []

        for execution in parsedlog:
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

    ## Generate features for each log file in the database. Take the index of the 40 most frequent words.
    def featurizeLF_db(self, w2v_dictionary, logid):
        words_to_write = 4
        kwsparse = []
        tfids = []

        model = w2v_dictionary
        print("\tThere are %d words in this model" % model.vocab.__len__())
        # if model.vocab.__len__() == 0:
        #     print('this model')
        try:
            most_frequent_words = model.index2word[0:words_to_write]
        except:
            most_frequent_words = model.index2word[0:model.vocab.__len__()]
        # print(most_frequent_words)

        for word in most_frequent_words:
            a = model[word]
            kwsparse.extend(a)
            b = model.most_similar(word)
            c = model[b[0][0]]
            tfids.extend(c)

        kwids, countids = self.dbwriter.getdictionarykeywordsbyid(most_frequent_words, words_to_write)
        # tfids.extend(self.count_to_tf(countids))

        kwsparse = np.asarray(kwsparse).astype(float).tolist()
        tfids = np.asarray(tfids).astype(float).tolist()

        self.dbwriter.writeLFfeatures(kwsparse, tfids, logid)

    def featurizeLF_testing(self, w2v_dictionary, words_to_write=4):
        features = []

        model = w2v_dictionary
        try:
            most_frequent_words = model.index2word[0:words_to_write]
        except:
            most_frequent_words = model.index2word[0:model.vocab.__len__()]

        for word in most_frequent_words:
            a = model[word]
            features.extend(a)
            b = model.most_similar(word)
            c = model[b[0][0]]
            features.extend(c)

        kwids, countids = self.dbwriter.getdictionarykeywordsbyid(most_frequent_words, words_to_write)
        # features.extend(self.count_to_tf(countids))

        features = np.asarray(features).astype(float)
        return features[0:80]

    def featurizeLI_db(self, w2v_dictionary, logid):
        features_to_write = 100

        model = w2v_dictionary
        print("\tThere are %d words in this model" % model.vocab.__len__())
        # if model.vocab.__len__() == 0:
        #     print('this model')
        try:
            most_frequent_words = model.index2word[0:features_to_write]
        except:
            most_frequent_words = model.index2word[0:model.vocab.__len__()]
        # print(most_frequent_words)
        kwids, tfids = self.dbwriter.getdictionarykeywordsbyid(most_frequent_words)
        self.dbwriter.writeLFfeatures(kwids, logid)

    def trainLFfeatures(self, queue=None, option=None):
        # Use a Naive Bayes classifier
        lf_nb_predictor = GaussianNB()

        lf_kmeans = KMeans(n_clusters=3, n_init=100)

        lf_mlp = MLPClassifier(hidden_layer_sizes=(10, 100, 1000, 100), activation='tanh', solver='sgd', batch_size=3,
                               learning_rate='adaptive', learning_rate_init=0.1, max_iter=5000, verbose=True,
                               shuffle=True)

        features, targetoutputs = self.dbwriter.getLFfeatures()
        # features[:, 0:40] = normalize(features[:, 0:40], axis=0)

        # lf_nb_predictor.fit(features, targetoutputs)
        # lf_kmeans.fit(features, targetoutputs)
        # lf_mlp.fit(features, targetoutputs)
        #
        # print("\tTraining Naive Bayes")
        # lf_nb_predictor.partial_fit(features, targetoutputs, classes=[1, 2, 3])
        # print("\tTraining K Means")
        # lf_kmeans.fit(features, targetoutputs)
        # print("\tTraining MLP Classifier")
        # lf_mlp.partial_fit(features, targetoutputs, classes=[1, 2, 3])

        if option is 1:
            msg = "\tTraining Naive Bayes"
            if queue is not None:
                queue.put(msg)
            else:
                print(msg)
            lf_nb_predictor.fit(features, targetoutputs)
            path_to_save = self.project_root + '/nb_models/nb'
            self.saveclassifier(path_to_save, lf_nb_predictor)

        if option is 2:
            msg = "\tTraining K Means"
            if queue is not None:
                queue.put(msg)
            else:
                print(msg)
            lf_kmeans.fit(features, targetoutputs)
            path_to_save_kmeans = self.project_root + '/nb_models/km'
            self.saveclassifier(path_to_save_kmeans, lf_kmeans)

        if option is 3:
            msg = "\tTraining MLP Classifier"
            if queue is not None:
                queue.put(msg)
            else:
                print(msg)
            lf_mlp.fit(features, targetoutputs)
            path_to_save_nn = self.project_root + '/nb_models/mlp'
            self.saveclassifier(path_to_save_nn, lf_mlp)

        return lf_nb_predictor

    def trainLFfeatures_BRBM(self):
        features, targetoutputs = self.dbwriter.getLFfeatures()
        # features[:, 0:40] = normalize(features[:, 0:40], axis=0)

        rbm = BernoulliRBM(verbose=True, batch_size=3, learning_rate=0.1, n_iter=10000, n_components=100)
        lf_mlp = MLPClassifier(hidden_layer_sizes=(80, 100, 100), activation='tanh', solver='sgd', batch_size=3,
                               learning_rate='adaptive', learning_rate_init=0.1, max_iter=5000, verbose=True,
                               shuffle=True)

        brbm_classifier = Pipeline([('rbm', rbm), ('mlp', lf_mlp)])

        print("\tTraining Bernoulli RBM network")

        brbm_classifier.fit(features, targetoutputs)
        pathtosave = self.project_root + '/nb_models/rbm'

        self.saveclassifier(path=pathtosave,
                            classifier=brbm_classifier)

        return brbm_classifier

    def randomize_data(self, x, y):
        random_indices = np.arange(y.__len__())
        np.random.shuffle(random_indices)

        randx = []
        randy = []

        for index in random_indices:
            randx.append(x[index])
            randy.append(y[index])

        return np.asarray(randx), np.asarray(randy)

    def trainSKNN(self):
        features, targetoutputs = self.dbwriter.getLFfeatures()
        features = np.asarray(features)
        targetoutputs = np.asarray(targetoutputs).reshape(-1, 1)

        randx, randy = self.randomize_data(features, targetoutputs)

        nn = Classifier(
            layers=[
                Layer("Rectifier", units=80),
                Layer("Sigmoid", units=20),
                Layer("Softmax", units=3)
            ],
            learning_rate=0.01,
            n_iter=100000,
            debug=True,
            verbose=True,
            batch_size=10)

        nn.partial_fit(randx, randy)

        pathtosave = self.project_root + '/nb_models/sknn'

        self.saveclassifier(pathtosave, nn)

    def saveclassifier(self, path, classifier):
        with open(path, 'wb') as pf:
            pickle.dump(classifier, pf)
        print("** Saved model to", path)

    def loadclassifier(self, path):
        with open(path, 'rb') as pf:
            cls = pickle.load(pf)
        return cls

    def testLFfeatures(self, queue):
        print("\nTesting trained models")
        path_to_saved_models = 'C:/logpredict_new-logpredict/nb_models/'
        models = os.listdir(path_to_saved_models)
        features, targetoutputs = self.dbwriter.getLFfeatures()
        targetoutputs = np.asarray(targetoutputs)

        # norm_features = normalize(features, axis=0)
        for model in models:
            try:
                if queue.queue[-1].find('teststop') is not -1:
                    queue.put("teststop")
                    break
            except Exception as e:
                # print("breaking", e)
                pass
            modelpath = path_to_saved_models + '/' + model
            model = self.loadclassifier(modelpath)
            a = ("\tTesting %s" % str(type(model)))
            queue.put(a)
            predicted = cross_val_predict(model, features, targetoutputs)
            print(a)
            print('\t\tAccuracy: ', accuracy_score(targetoutputs, predicted))
            print('\t\tPrecision: ', precision_score(targetoutputs, predicted, average='weighted'))
            queue.put(predicted)
            # fig, ax = plt.subplots()
            # ax.scatter(targetoutputs, predicted)
            # ax.plot([0, 3], [0, 3], 'k--', lw=1)
            # ax.set_xlabel('Measured')
            # ax.set_ylabel('Predicted')
            # plt.title(str(type(model)))
            # plt.show()

    def test(self):
        self.dbwriter.test()

    def setupLFtestdictionary(self, parsedlog):
        num_features = 4
        stf, tm, clf = self.sanitizelogfile_object(parsedlog)

        testids = self.featurizeLF_testing(tm, num_features)
        return testids

    def testnewlogfiles(self, parsedlog, file):
        testfeatures = self.setupLFtestdictionary(parsedlog)
        predictions = {}
        print("Testing %s" % file)
        # model_to_load = self.project_root + '/nb_models/nn'
        # nbc = self.loadclassifier(model_to_load)

        path_to_saved_models = self.project_root + '/nb_models/'
        models = os.listdir(path_to_saved_models)

        testfeatures = np.asarray(testfeatures).reshape(1, -1)

        # norm_features = normalize(features, axis=0)
        for model in models:
            modelpath = path_to_saved_models + '/' + model
            model = self.loadclassifier(modelpath)
            # print("\tUsing %s" % str(type(model)))
            try:
                # For BernoulliRBM model
                prediction = model.score_samples(testfeatures)
            except:
                prediction = model.predict(testfeatures)
                #
                # print('\t', prediction),
                # if prediction.any() < 1:
                #     print('\t', file, 'Debug SCA log')
                # elif prediction.any() < 2:
                #     print('\t', file, 'SCA log')
                # elif prediction.any() < 3:
                #     print('\t', file, 'SSC log')
            predictions[str(type(model))] = prediction
        for i, p in predictions.items():
            print(i, p)
        return predictions
