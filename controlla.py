from classes import Logfile
import fstorage
import os
from db_trainer import db_trainer
import threading
import numpy as np
import time


class ThreadedTask_parser(threading.Thread):
    def __init__(self, queue, flag):
        threading.Thread.__init__(self, args=flag)
        self.flag = flag
        self.queue = queue
        self.controlla = controlla()

    def run(self):
        controlla.parselogs(self.controlla, queue=self.queue, flag=self.flag)


class ThreadedTask_trainer(threading.Thread):
    def __init__(self, queue, options):
        threading.Thread.__init__(self)
        self.queue = queue
        self.controlla = controlla()
        self.options = options

    def run(self):
        controlla.setupdictionary(self.controlla, queue=self.queue, options=self.options)


class ThreadedTask_tester(threading.Thread):
    def __init__(self, queue, flag):
        threading.Thread.__init__(self, args=flag)
        self.flag = flag
        self.queue = queue
        self.controlla = controlla()

    def run(self):
        controlla.testLFfeaturesmodels(self.controlla, queue=self.queue)


class ThreadedTask_predictor_single_file(threading.Thread):
    def __init__(self, queue, path):
        threading.Thread.__init__(self)
        self.path = path
        self.queue = queue
        self.controlla = controlla()

    def run(self):
        predictions = controlla.testnew_singlelog(self.controlla, path=self.path, queue=self.queue)
        self.queue.put(predictions)


class controlla:
    fs = fstorage.featurestorage()
    dbt = db_trainer()

    def __init__(self):
        # Get the path of the current working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.abspath(os.curdir)

    def parselogs(self, flag, queue, logfilestoadd=1000):
        logroot = self.project_root + '/training_logs/'
        logtoparse = Logfile.Logfile()
        fs = fstorage.featurestorage()
        products = os.listdir(logroot)
        breakcount = 0

        while breakcount < logfilestoadd:
            try:
                if queue.queue[-1].find('Stopped') is not -1:
                    queue.put("Stopped Thread - master loop")
                    break
            except Exception as e:
                pass

            for product in products:
                try:
                    if queue.queue[-1].find('Stopped') is not -1:
                        queue.put("Stopped Thread - product loop")
                        break
                except Exception as e:
                    pass
                print("\n\nParsing %s logs" % product)
                product_path = (logroot + '/' + product)
                files = os.listdir(product_path)
                if 'category' in files:
                    dir_path = product_path + '/' + 'category'
                    categories = os.listdir(dir_path)
                    for category in categories:
                        if breakcount == logfilestoadd:
                            break
                        try:
                            if queue.queue[-1].find('Stopped') is not -1:
                                queue.put("Stopped Thread - category loop")
                                break
                        except Exception as e:
                            pass
                        print("\n\tFrom category %s..." % category)
                        dir_category = dir_path + '/' + category
                        files = os.listdir(dir_category)

                        for file in files:
                            try:
                                if queue.queue[-1].find('Stopped') is not -1:
                                    queue.put("Stopped Thread - file category loop")
                                    break
                            except Exception as e:
                                pass

                            print("\t\tLogs left to parse: ", logfilestoadd - breakcount)
                            print('\t\t{0:2}. {1:50}'.format(breakcount + 1, file), end='....... ')
                            filetoopen = (dir_category + '/' + file)
                            parsedlog = logtoparse.parseFile(filetoopen)
                            parsedlog.scantype = product
                            try:
                                fs.commit_db(parsedlog)  # Write logfile to the database
                                print("Success")
                            except:
                                print("FAILED")
                            breakcount += 1
                            if breakcount == logfilestoadd:
                                break

                else:
                    print("No categories set, parsing the following files at LOGFILE level ONLY")
                    for file in files:
                        try:
                            if queue.queue[-1].find('Stopped') is not -1:
                                queue.put("Stopped Thread - Logfile level loop")
                                break
                        except:
                            continue
                        print('\t\t{0:2}. {1:50}'.format(breakcount + 1, file), end='....... ')
                        filetoopen = (product_path + '/' + file)
                        parsedlog = logtoparse.parseFile(filetoopen)
                        parsedlog.scantype = product
                        self.fs.commit_db(parsedlog)
                        print("Success")
                        breakcount += 1
                        if breakcount == logfilestoadd:
                            break

    def setupdictionary(self, queue, options):
        self.dbt.generatedictionary(queue)
        modelstotrain = []
        for option in options:
            if option.get() is 1:
                if option._name == 'RBM':
                    if queue.queue[-1].find('trainstop') is -1:
                        a = "\tTraining BernoulliRBM Neural Net"
                        queue.put(a)
                        self.dbt.trainLFfeatures_BRBM()
                if option._name == 'MLP':
                    # Don't need to put anything into the queue since trainLFfeatures() will do it
                    if queue.queue[-1].find('trainstop') is -1:
                        self.dbt.trainLFfeatures(queue, option=3)
                if option._name == 'KM':
                    # Don't need to put anything into the queue since trainLFfeatures() will do it
                    if queue.queue[-1].find('trainstop') is -1:
                        self.dbt.trainLFfeatures(queue, option=2)
                if option._name == 'NN':
                    # Don't need to put anything into the queue since trainLFfeatures() will do it
                    if queue.queue[-1].find('trainstop') is -1:
                        self.dbt.trainLFfeatures(queue)
                if option._name == 'NB':
                    # Don't need to put anything into the queue since trainLFfeatures() will do it
                    if queue.queue[-1].find('trainstop') is -1:
                        self.dbt.trainLFfeatures(queue, option=1)
                if option._name == 'SKNN':
                    if queue.queue[-1].find('trainstop') is -1:
                        a = "\tTraining SKNN Neural Net"
                        queue.put(a)
                        self.dbt.trainSKNN()









        # try:


    def testLFfeaturesmodels(self, queue):
        self.dbt.testLFfeatures(queue)

    # Start training on the logfiles in the database
    def trainLF(self, threadstop):
        # classifier = self.dbt.trainLFfeatures()
        self.dbt.trainSKNN()
        # self.dbt.trainLFfeatures()
        # self.dbt.trainLFfeatures_BRBM()
        # self.dbt.testLFfeatures()

    def test(self):
        self.dbt.test()

    # Test trained models on new logs
    def testnewlogs(self):
        logtoparse = Logfile.Logfile()
        path_to_test = self.project_root + '/test_logs/'
        files = os.listdir(path_to_test)

        for file in files:
            path_to_file = path_to_test + file
            parsedlog = logtoparse.parseFile(path_to_file)
            self.dbt.testnewlogfiles(parsedlog, file)

    def testnew_singlelog(self, path, queue):
        logtoparse = Logfile.Logfile()
        parsedlog = logtoparse.parseFile(path)
        predictions = self.dbt.testnewlogfiles(parsedlog, path)
        return predictions

    # Start program
    def run(self, threadstop):
        self.parselogs()
        self.setupdictionary(threadstop)
        self.trainLF(threadstop)

# #
# #
# ct = controlla()
# # # ct.test()
# # ct.run()
# ct.testnewlogs()
