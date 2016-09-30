from time import time

from classes import Execution
from classes import Logitem
from classes import Logfile
from classes import LogException
from classes import Stacktrace
from classes import Parser
from scipy import *
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
import os
import ssclog_learner
import nn3_logfile
import sscexecution_learner
import fstorage
import ssclog_learner
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import linear_model
import fstorage

__author__ = 'Sourabh'
#### PARSE LOG FROM LOG > EXECUTIONS > LOGITEMS

#### Get LOG TYPE
## Parse out log level features
## NN1 to train features and product category
## NN1 output = product type

#### Get EXECUTION TYPE
## Parse out execution level features
## NN2 to train on features and their product category
## NN2 output = execution category type

#### Get LOGITEM SCORE
## Parse out logitem level features
## NN3 to train on features and their logitem score
## NN3 output = highest scoring logitem

#### Train LOG TYPE => LOGITEM SCORE
## Merge each logitem feature with that of its log file + NN1 output
## Add additional features (?)
## NN4 to train on features and their logitem score
## NN4 output =

# Run initial classifier to find out which type of log it is

logtoparse = Logfile.Logfile()

nn3 = nn3_logfile.parser()
nn2 = ssclog_learner.SSC_nn()
nn1 = sscexecution_learner.ssc_logitem_nn()

fs = fstorage.featurestorage()

logfilestoadd = 100

logitem_rbm_predictor = BernoulliRBM(n_components=10, n_iter=10, learning_rate=0.1, batch_size=5, verbose=1)
logfile_rbm_predictor = BernoulliRBM(n_components=200, n_iter=10, learning_rate=0.5, batch_size=3, verbose=0)

logitem_reg_predictor = linear_model.LogisticRegression(C=100.0)
classifier = Pipeline(steps=[('rbm', logitem_rbm_predictor), ('logistic', logitem_reg_predictor)])

parsers = ["ssc"]
repeatepoch = True
while repeatepoch:

    ## Determine which phase of parser we are at and fetch next training log
    # 1. Get log file features
    training_logs_root = "C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/training_logs/"
    products = os.listdir(training_logs_root)
    logfilenum = 0
    breakcount = 0
    while breakcount < logfilestoadd:

        for product in products:
            print("\n\nParsing %s logs" % product)
            product_path = (training_logs_root + '/' + product)
            files = os.listdir(product_path)

            if product == 'ssc':
                nn3_target_output = 1
            elif product == 'sca':
                nn3_target_output = 2
            elif product == 'dsca':
                nn3_target_output = 3

            if 'category' in files:
                dir_path = product_path + '/' + 'category'
                categories = os.listdir(dir_path)
                for category in categories:
                    if breakcount == logfilestoadd:
                        break
                    print("\tFrom category %s..." % category)
                    dir_category = dir_path + '/' + category
                    files = os.listdir(dir_category)

                    for file in files:
                        print("Logs left to parse: ", logfilestoadd - breakcount)
                        print('{0:2}. {1:50}'.format(breakcount + 1, file), end='....... ')
                        filetoopen = (dir_category + '/' + file)
                        parsedlog = logtoparse.parseFile(filetoopen)

                        # # Get logfile features
                        # feature3 = nn3.parse(filetoopen, product)
                        # fs.add_logfile(feature3, file)
                        #
                        # # Get execution features
                        # feature2 = nn2.extractFeatures(parsedlog, category, file)
                        # fs.add_executions(feature2)
                        #
                        # # Get logitem features
                        # feature1 = nn1.extractTrainingFeatures(parsedlog)
                        # fs.add_logitems(feature1)

                        # fs.commit(parsedlog)
                        fs.commit_db(parsedlog)

                        print("Success")

                        #                   execcount = parsedlog.getExecutions().__len__()
                        #                   for execution in parsedlog.getExecutions():
                        #                       logcount=execution.getlogitemlength()
                        #                       print("\tTotal number of log items in this execution: %d" % logcount)

                        try:
                            test1 = np.asarray(feature2)
                            print("\tLogfile data size: (1,%d)" % feature3.__len__())
                            print("\tExecution data size: ", test1.shape)
                            print("\tLogitem data size: ", nn1.getShape(feature1))
                        except:
                            continue

                        breakcount += 1
                        if breakcount == logfilestoadd:
                            break
            else:
                print("No categories set, parsing the following files at LOGFILE level ONLY")
                fc = 0
                for file in files:
                    dir_path = (product_path + '/' + file)

                    print('{0:2}. {1:50}'.format(fc + 1, file), end='....... ')
                    try:
                        feature1 = nn3.parse(parsedlog=dir_path, product=product)

                        feature3 = np.asarray(feature1)
                        x = feature3.shape[0] - 1
                        training_features = feature3[:x]
                        training_output = feature3[x]

                        x = logfile_rbm_predictor.fit_transform(training_features, training_output)
                        print("Success")

                    except Exception as e:
                        print(" ... Fail", e.message)
                    fc += 1

    m1 = fs.getMergedtable()
    print("Total merged data of shape [%d, %d]" % (m1.__len__(), m1[0].__len__()))
    # m0 = fs.getMergedtable(0)
    # print ("m0 shape [%d, %d]" % (m0.__len__(), m0[0].__len__()))

    # lifeatures = np.asarray(fs.logitem_features.values())
    # test = fs.logitem_features.values()
    lifeatures = []
    i = 0
    test1 = list(fs.logitem_features.values())
    test1 = np.asarray(test1)
    test1 = test1.astype(float)
    # {lifeatures[i]:test1[i] for i in range(len(lifeatures))}
    print("Total logitem features shape: ", test1.shape)

    exfeatures = list(fs.execution_features.values())
    exfeatures = np.asarray(exfeatures)
    print("Total execution features shape: ", exfeatures.shape)

    lffeatures = list(fs.logfile_features.values())
    lffeatures = np.asarray(lffeatures)
    print("Total logfile features shape: ", lffeatures.shape)

    print('\n\n')
    print("##########################################################################################")
    print("Predictions")
    print("##########################################################################################")

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    ###############################   Logitem predictions   ##################################

    # Get logitem table
    raw_data = np.asarray(m1)
    raw_data = raw_data.astype(float)

    # Setup logitem training, testing data
    training_features, training_output, test_features = fs.getTables(raw_data)
    test_output = training_output[:2]

    training_output = np.asarray(training_output)
    training_features = np.asarray(training_features).astype(float)

    # Set up logitem predictor
    # logitem_predictor = BernoulliRBM(n_components=20, n_iter=100, learning_rate=1.5, batch_size=3, verbose=1)
    print(logitem_rbm_predictor.get_params())

    # Fit the training/testing data to the predictor
    t1 = time()
    # g = logitem_rbm_predictor.fit_transform(training_features, training_output)
    # logitem_reg_predictor.fit(np.asarray(training_features).astype(float), np.asarray(training_output).astype(float))

    # logitem_reg.fit(training_features, test_features)
    logitem_rbm_predictor.fit_transform(training_features, training_output)
    logitem_reg_predictor.fit(training_features, training_output)

    ###############################   Execution predictions   ##################################

    # Setup execution table
    raw_data = exfeatures

    # Setup execution training, testing data
    training_features, training_output, test_features = fs.getTables(raw_data)
    test_output = training_output[:2]
    training_output = np.asarray(training_output)
    training_features = np.asarray(training_features)
    training_features.astype(float)

    # Setup execution predictor
    logistic2 = linear_model.LinearRegression()
    execpreds = logistic2.fit(training_features, training_output)

    ###############################   Logfile predictions   ##################################

    # Setup logfile table
    raw_data = lffeatures

    # Setup logfile training, testing data
    training_features, training_output, test_features = fs.getTables(raw_data)
    test_output = training_output[:2]
    training_output = np.asarray(training_output)
    training_features = np.asarray(training_features)
    training_features.astype(float)

    # Setup logfile predictor
    logfiletype_classifier = linear_model.LinearRegression()
    logfiletype_classifier.fit(training_features, training_output)
    logfile_rbm_predictor.fit_transform(training_features, training_output)

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    ##############################   Test by parsing a log   #################################

    print('##########################################################################################')
    print("\nStarting to parse a test log")
    testlogs_path = "C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/test_logs"
    files = os.listdir(testlogs_path)

    if files.__len__() > 0:
        for file in files:
            fs2 = fstorage.featurestorage()
            print('Parsing', file)
            filetoopen = (testlogs_path + '/' + file)
            parsedlog = logtoparse.parseFile(filetoopen)

            # Get logfile features
            feature3 = nn3.parse(filetoopen, 'test')
            feature3 = np.asarray(feature3)
            try:
                x, y = feature3.shape
            except:
                y = feature3.shape[0]
            feature31 = feature3[:y - 2]
            test = np.asarray(logfile_rbm_predictor.score_samples(feature31)).astype(float)

            if test <= 1.0:
                fs2.add_logfile(feature3, file)
                parsedlog = logtoparse.parseFile(filetoopen)

                # Get execution features
                feature2 = nn2.extractFeatures(parsedlog, 'test', file)
                feature12 = np.asarray(feature2).astype(float)
                feature2f = []
                try:
                    feature2_np = np.asarray(feature2)
                    x, y = feature2_np.shape
                except:
                    y = feature2.shape
                for row in feature2[:1]:
                    feature2f.append(row[:y - 2])
                categoriess = np.asarray(logistic2.predict(feature2f)).astype(float)

                fs2.add_executions(feature2)

                # Get logitem features
                feature1 = nn1.extractTrainingFeatures(parsedlog)
                feature1 = np.asarray(feature1).astype(float)
                fs2.add_logitems(feature1)
                # fs2.commit(parsedlog)
                fs2.commit_db(parsedlog)

            elif test == 2.0:
                print("Most probably an SCA log")
                continue

            elif test == 3.0:
                print("Most probably a debug SCA log")
                continue

            mergedz = fs2.getMergedtable()
            trainf, traino, testf = fs2.getTables(np.asarray(mergedz))

            mergedf = np.asarray(mergedz).astype(float)
            mergedff = []
            try:
                x, y = mergedf.shape
                print(mergedf.shape)
            except:
                y = mergedf.shape[0]
                row = np.asarray(mergedf).astype(float)
                mergedff.append(row[:(y - 2)])
            for i in range(0, x):
                row = mergedf[i].copy()
                row = np.asarray(row).astype(float)
                mergedff.append(row[:(y - 2)])

            preds = np.asarray(logitem_rbm_predictor.score_samples(trainf)).astype(float)
            test2 = np.asarray(logitem_reg_predictor.predict(trainf)).astype(float)

            fs2.getpreds(test, 'logfile')
            fs2.getpreds(categoriess, 'execution')
            lix = fs2.getpreds({'rbmpreds': preds, 'regpreds': test2}, 'logitem')
            # lix1 = fs2.getpreds(test2, 'logitem')


            # choices = input("Are any of these logitems correct? ")
            choices = 'no'
            if choices == 'no':
                logitem_rbm_predictor.learning_rate += 0.1
                logitem_rbm_predictor.batch_size += 5
                continue
            elif choices.__len__() > 0:
                fs.add_user_choices(choices, lix)
            else:
                print("Not a valid input")

    valid = True
    while not valid:
        repeatchoice = input("Repeat for another epoch? ")
        if repeatchoice == 'yes':
            repeatepoch = True
            valid = True
        elif repeatchoice == 'no':
            repeatepoch = False
            valid = True
        else:
            print("Not a valid input")
