from classes import Logfile
import fstorage
import os
from db_trainer import db_trainer


class controlla:
    fs = fstorage.featurestorage()
    dbt = db_trainer()

    def __init__(self):
        pass

    def parselogs(self, logroot="C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/training_logs/",
                  logfilestoadd=100):
        logtoparse = Logfile.Logfile()
        fs = fstorage.featurestorage()
        products = os.listdir(logroot)
        logfilenum = 0
        breakcount = 0
        while breakcount < logfilestoadd:
            for product in products:
                print("\n\nParsing %s logs" % product)
                product_path = (logroot + '/' + product)
                files = os.listdir(product_path)

                if 'category' in files:
                    dir_path = product_path + '/' + 'category'
                    categories = os.listdir(dir_path)
                    for category in categories:
                        if breakcount == logfilestoadd:
                            break
                        print("\n\tFrom category %s..." % category)
                        dir_category = dir_path + '/' + category
                        files = os.listdir(dir_category)

                        for file in files:
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
                        print('\t\t{0:2}. {1:50}'.format(breakcount + 1, file), end='....... ')
                        filetoopen = (product_path + '/' + file)
                        parsedlog = logtoparse.parseFile(filetoopen)
                        parsedlog.scantype = product
                        self.fs.commit_db(parsedlog)
                        print("Success")
                        breakcount += 1
                        # self.dbt.featurizeLF(parsedlog)

                        # execs = parsedlog.getExecutions()
                        # for execa in execs:
                        #     print("\t\t\tExecution")
                        #     lii = execa.getlogitems()
                        #     print("\t\t\t\t%d logitems" % lii.__len__())

    def setupdictionary(self):
        self.dbt.generatedictionary()

    def trainLF(self):
        classifier = self.dbt.trainLFfeatures()
        self.dbt.testLFfeatures(classifier)

    def test(self):
        self.dbt.test()

    def testnewlogs(self):
        logtoparse = Logfile.Logfile()
        path_to_test = 'C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/test_logs/'
        files = os.listdir(path_to_test)
        nbc = self.dbt.loadclassifier("C:/Users/Sourabh/Documents/logpredict/logpredict/logpredict/nb_models/nb1")

        for file in files:
            path_to_file = path_to_test + file
            parsedlog = logtoparse.parseFile(path_to_file)
            testfeatures = self.dbt.setupLFtestdictionar(parsedlog)
            prediction = nbc.predict(testfeatures)

            print('\n')
            if prediction == 1:
                print(file, 'Debug SCA log')
            elif prediction == 2:
                print(file, 'SCA log')
            elif prediction == 3:
                print(file, 'SSC log')

    def run(self):
        # self.parselogs()
        self.setupdictionary()
        self.trainLF()


ct = controlla()
ct.run()
ct.testnewlogs()
