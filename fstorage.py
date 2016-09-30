__author__ = 'SourabhKatti'
import array
from classes import Logfile
import numpy as np
from storage import storage


class featurestorage:
    index_mappings = {}
    exec_index_mappings = {}

    logfile_features = {}
    execution_features = {}
    logitem_features = {}

    lf_index = 0
    exec_index = 0
    li_index = 0

    merged = []

    logfiles = {}

    filenames = {}

    def __init__(self):

        self.index_mappings = {}
        self.exec_index_mappings = {}

        self.logfile_features = {}
        self.execution_features = {}
        self.logitem_features = {}

        self.lf_index = 0
        self.exec_index = 0
        self.li_index = 0

        self.merged = []

        self.logfiles = {}

        self.filenames = {}

        self.dbwriter = storage.db_store()

    ## Append each line with its execution features and its logfile features
    def getMergedtable(self):
        self.merged.clear()
        print("\nGetting merged table..")
        for index in range(0, self.logitem_features.__len__()):
            lf_index, exec_index, li_index = self.index_mappings[index]

            if lf_index > self.logfile_features.__len__():
                break
            if exec_index > self.execution_features.__len__():
                break
            if li_index > self.logitem_features.__len__():
                break
            lff = self.logfile_features.get(lf_index - 1).copy()
            execf = self.execution_features.get(exec_index - 1).copy()
            lif = self.logitem_features.get(index)

            try:
                execf.extend(lif)
            except:
                execf = np.append(execf, lif)
            try:
                lff.extend(execf)
            except:
                lff = np.append(lff, execf)
            self.merged.append(np.asarray(lff, dtype=np.uint8).ravel())
        return self.merged

    def addlogfiletorepository(self, logfile, index):
        self.logfiles[index] = logfile

    def getlogfilebyindex(self, index):
        lf_index, exec_index, li_index = self.index_mappings.get(index)
        logfile = self.logfiles.get(lf_index)
        filename = self.filenames.get(lf_index)
        return logfile, filename

    def getlifeatures(self):
        featurestoreturn = []
        for row in self.logitem_features:
            rowtoreturn = list(row)

    def getexecutionbyindex(self, index):
        lf_index, exec_index, li_index = self.index_mappings[index]
        logfile = self.logfiles.get(lf_index)
        i = 0
        executionret = logfile.getExecutions()[exec_index]
        return executionret

    def getlogitembyindex(self, index):
        lf_index, exec_index, li_index = self.index_mappings[index]
        logfile = self.logfiles.get(lf_index)
        i = 0
        executionf = logfile.getExecutions()[exec_index]
        logitemf = executionf.logitems[li_index]
        return logitemf
        print("No execution for that index")

    def commit(self, logfile):
        li_old = self.li_index
        lf_old = self.lf_index + 1
        exec_old = self.exec_index

        for execution in logfile.getExecutions():
            exec_old += 1
            logitem_index = 0
            try:
                for logitem in execution.getlogitems():
                    value = [lf_old, exec_old, logitem_index]
                    self.index_mappings[li_old] = value
                    li_old += 1
                    logitem_index += 1
                self.addlogfiletorepository(logfile, lf_old)
            except:
                print("Indexing gone past logitem feature list, logfile may not match")

        return self.index_mappings

    def commit_db(self, logfile):

        lf_db_index = self.dbwriter.write_logfile(logfile)

        for execution in logfile.getExecutions():
            exec_db_index = self.dbwriter.write_execution(execution, lf_db_index)
            li_db_indices = self.dbwriter.write_logitems(execution.getlogitems(), exec_db_index)

    def getdictionarywordsbykeyword(self, keywords):
        self.dbwriter.getdictionarykeywordsbyid(keywords)


    def add_logitems(self, features):
        self.li_index = self.logitem_features.__len__()
        current_index = self.li_index
        li_len = features.__len__()
        for row in features:
            self.logitem_features[current_index] = row
            current_index += 1

    def add_executions(self, features1):
        self.exec_index = self.execution_features.__len__()
        current_index = self.li_index
        exec_length = 0
        for feature in features1:
            self.execution_features[self.exec_index + exec_length] = feature
            exec_length += 1

    def add_logfile(self, features, filename):
        self.lf_index = self.logfile_features.__len__()
        self.filenames[self.lf_index] = filename
        self.logfile_features[self.lf_index] = features

    def getlogfilebytableindex(self, index):
        lf = self.logfiles.get(index)
        return lf

    def getTables(self, mergedtable):
        train_features = []
        train_class = []

        test_features = []
        fcount = 0

        for row in mergedtable:
            lenf = row.__len__() - 1
            f = row[:lenf - 1]
            train_features.append(f)
            train_class.append(row[lenf])

            if fcount < 3:
                test_features.append(row[:lenf - 1])
                fcount += 1

        return train_features, train_class, test_features

    def getpreds(self, values, level):

        if level == 'logitem':
            li = []
            index = 0
            maxz = min(values['rbmpreds'])
            for pred in values['rbmpreds']:

                if pred < 0.6 * maxz:
                    lf, execi, lii = self.index_mappings[index]

                    logfile = self.logfiles[lf]
                    try:
                        execution = logfile.getExecutions()[execi - 1]
                    except:
                        execution = logfile.getExecutions()

                    try:
                        logitem = execution.getlogitems()[lii - 1]
                    except:
                        logitem = execution.getlogitems()

                    li.append({'pred': pred, 'logitem': logitem})

                index += 1
            linum = 1

            lix = sorted(li, key=lambda k: k['pred'])

            print("\nPossible logitems: ")
            linum = 1
            body = ""
            for logitemscore in lix:
                logitem = logitemscore['logitem']
                print("%d. %s on line %d" % (linum, logitem.methodname, logitem.linenumber))
                body += "%d. %s at line number %d with a score of %f.\n" % (
                linum, logitem.methodname, logitem.linenumber, logitemscore['pred'])
                linum += 1

            title = "Predictions for logfile %d" % linum

            return title, body


        elif level == 'execution':
            execs = []
            index = 0
            print('\nExecution category')
            i = 0
            for value in values:
                if int(value) == 1:
                    print("%s. Most probably an issue with your application server" % str(i + 1))
                elif int(value) == 2:
                    print("%s. Most probably an issue with your database" % str(i + 1))
                elif int(value) == 3:
                    print("%s. Most probably an issue with your LDAP server" % str(i + 1))
                else:
                    print("Invalid prediction", value)
                i += 1

        elif level == 'logfile':
            lfs = []
            index = 0
            print("\nLogfile type")
            try:
                for value in values:
                    try:
                        logfile = self.logfiles[index + 1]
                        lfs.append(logfile)
                    except:
                        break

                    if int(value) <= 1:
                        print("%s is most probably an SSC log" % self.filenames[index])
                    elif int(value) == 2:
                        print("%s is most probably an SCA log" % self.filenames[index])
                    elif int(value) == 3:
                        print("%s is most probably a debug SCA log" % self.filenames[index])
                    else:
                        print("Invalid predictions", value)

                    index += 1
            except:

                if int(values) <= 1:
                    print("%s is most probably an SSC log" % self.filenames[index])
                elif int(values) == 2:
                    print("%s is most probably an SCA log" % self.filenames[index])
                elif int(values) == 3:
                    print("%s is most probably a debug SCA log" % self.filenames[index])
                else:
                    print("Invalid predictions", value)

                index += 1
