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


class SSC_nn:
    counts = 0.0
    log_features = []
    class_guess = []
    test_classguess = []
    testing_features = []
    java_classes = {}
    system_properties = {}
    testfile_index = {}
    ssc_features = {}
    execweights = {}
    test_execution_weights = {}

    categories = ['tomcat', 'catalina', 'mysql', 'MysqlParameterMetadata', 'com.mysql.jdbc.SQLError.createSQLException',
                  "memory", "GC", "resources", "MaxPermSize", 'packet', 'packet sent', 'received any packets',
                  'FortifyException', 'ConnectException', 'Connection', 'timed', 'connect',
                  'java.net.DualStackPlainSocketImpl.connect0(Native Method)', 'Connection timed out', 'accessible',
                  'not accessible',
                  'not down', 'overloaded', 'Connection refused', 'RulepackBLLImpl',
                  'com.fortify.manager.BLL.impl.RulepackBLLImpl', 'importinstalledrulepacks',
                  'Exception importinstalledrulepacks',
                  'FortifyException', 'security content', 'certpath', 'sun.security.validator.ValidatorException',
                  'ValidatorException', 'PKIX', 'BugTrackerException',
                  'com.fortify.pub.bugtracker.support.BugTrackerException', 'SSLHandshakeException',
                  'SunCertPathBuilderException', 'certification', 'valid certification path', 'PKIXValidator', 'HY000',
                  '.SqlExceptionHelper', 'SQL Error', 'database', 'seeding', 'Migration', 'assertion', 'Hibernate',
                  'com.microsoft.sqlserver.jdbc.SQLServerException', 'SQLServerException',
                  'org.hibernate.exception.constraintviolationexceptioncould', 'hibernatedatabaseinterface.java',
                  'sqlexceptionhelper.java', 'database', 'hibernate', 'DB2', 'mysql', 'SQLException',
                  'com.fortify.manager.service.scheduler.SchedulerManagerImpl',
                  'org.hibernate.engine.jdbc.spi.SqlExceptionHelper', 'transaction', 'JDBCConnectionException',
                  'SQLState', 'RuntimeControllerConnectionConfiguration',
                  'Initializing runtime', 'event handlers',
                  'com.fortify.manager.service.runtime.RuntimeControllerConnectionConfiguration',
                  'Initializing runtime event handlers', '10234',
                  'com.fortify.manager.service.runtime.RuntimeControllerConnectionConfiguration', 'certification',
                  'unable to find valid certification path to requested target',
                  'sun.security.provider.certpath.SunCertPathBuilderException',
                  'sun.security.provider.certpath.SunCertPathBuilderException',
                  'sun.security.validator.ValidatorException',
                  'org.apache.xmlrpc.XmlRpcException', 'javax.net.ssl.SSLHandshakeException',
                  'java.security.cert.CertificateException', 'Bugzilla',
                  'com.fortify.pub.bugtracker.support.BugTrackerAuthenticationException',
                  'com.fortify.pub.bugtracker.support.BugTrackerAuthenticationException', 'Bugzilla4BugTrackerPlugin',
                  'ScanProcessAuthorizationException', 'authorization', 'REQUIRE_AUTHORIZATION', 'upload artifact',
                  'ArtifactUploadJob', 'SessionTimeoutFilter', 'upload', 'port', '10234', 'UserError',
                  'com.fortify.manager.service.parser.checker.ScanProcessAuthorizationException', 'Authentication',
                  'AuthenticationSuccessEvent', 'Authentication event AuthenticationSuccessEvent: admin',
                  'org.eclipse.birt.report.engine.api.impl.reportengine', 'org.eclipse.birt.report.engine.layout.html',
                  'org.eclipse.birt.data.engine.odaconsumer', 'parameter', 'birt', 'ReportBLLImpl', 'report',
                  'report generation', 'jobCallback', 'Trigger', 'PV$', 'scheduler', 'Quartz',
                  'com.fortify.manager.BLL.jobs.cron.RecurringLdapRefreshJob', 'job scheduler', 'job recovery',
                  'NameNotFoundException', 'ldap', 'DirectLdapObjectSource',
                  'com.fortify.manager.service.ldap.impl.DirectLdapObjectSource',
                  'javax.naming.CommunicationException', 'simple bind', 'severe', 'fatal', 'bind',
                  'convertLdapException', 'LdapClient', 'sun.security.ssl', 'sun.security.provider', 'SQLCODE', 'DB2',
                  'DB', 'SQLRecoverableException',
                  'Upload artifact', 'artifact failed', 'quartz', 'com.microsoft.sqlserver', 'bad SQL',
                  'FMUserInputException', 'CurrentStateFprDownload']

    def vectorizeWords(self, execution):
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\S+', min_df=1)
        x = vectorizer.fit_transform(execution)
        self.counts = x.toarray()
        # print ("no words")
        return vectorizer

    def getTF(self):
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(self.counts)
        return tfidf

    def getJavaFeatures(self, java_type):
        if java_type in self.java_classes.keys():
            return self.java_classes[java_type]
        else:
            count = self.java_classes.__len__() + 1
            self.java_classes[java_type] = count
            return count

    def getSystemFeatures(self, system_property):
        if system_property in self.system_properties.keys():
            return self.system_properties[system_property]
        else:
            count = self.java_classes.__len__() + 1
            self.system_properties[system_property] = count
            return count

    def getSSCVersion(self, ssc_feature):
        if ssc_feature in self.ssc_features.keys():
            return self.ssc_features[ssc_feature]
        else:
            count = self.ssc_features.__len__() + 1
            self.ssc_features[ssc_feature] = count
            return count

    def setCustomFeatures(self, vectorizer):
        bagofwords = vectorizer.get_feature_names()
        feature1 = '0'
        feature2 = '0'
        feature3 = '0'
        feature4 = '0'
        feature5 = '0'
        feature6 = '0'
        feature7 = '0'
        # print bagofwords

        tfidf = self.getTF()
        prob_max = 0.0
        cat_max = ""
        featurelist = []
        for word in self.categories:
            count = 0
            keywordcount = 0
            for logword in bagofwords:
                count = logword.count(word)
                keywordcount += count
            if keywordcount > 0:
                index = vectorizer.vocabulary_.get(logword)
                a = tfidf[:, index]
                featurelist.append((count + 1) * sum(a.data))
                if sum(a.data) > prob_max:
                    prob_max = sum(a.data)
            else:
                featurelist.append(0.0)

        jvmverspattern = 'java.runtime.version=(.*)'
        jvtypepattern = 'java.vm.specification.vendor=(\w+)'

        osarch = 'os.arch=(.*)'
        osname = 'os.name=(.*)'
        osbit = 'sun.arch.data.model=(.*)'
        ospattern = 'OperatingSystemMXBean: (.*)'

        sscversionpattern = 'Version:([\d\.]+)'

        jv = re.compile(jvmverspattern, re.IGNORECASE | re.DOTALL)
        jvt = re.compile(jvtypepattern, re.IGNORECASE | re.DOTALL)

        osa = re.compile(osarch, re.IGNORECASE | re.DOTALL)
        osn = re.compile(osname, re.IGNORECASE | re.DOTALL)
        osb = re.compile(osbit, re.IGNORECASE | re.DOTALL)
        osp = re.compile(ospattern, re.IGNORECASE | re.DOTALL)

        sscv = re.compile(sscversionpattern, re.IGNORECASE | re.DOTALL)

        for word in bagofwords:
            jv_match = jv.match(word)
            jvt_match = jvt.match(word)
            osa_match = osa.match(word)
            osn_match = osn.match(word)
            osb_match = osb.match(word)
            osp_match = osp.match(word)
            sscv_match = sscv.match(word)

            if jv_match:
                feature1 = (self.getJavaFeatures(jv_match.group(1)))

            if jvt_match:
                feature2 = (self.getJavaFeatures(jvt_match.group(1)))

            if osa_match:
                feature3 = (self.getSystemFeatures(osa_match.group(1)))

            if osn_match:
                feature4 = (self.getSystemFeatures(osn_match.group(1)))

            if osb_match:
                feature5 = (self.getSystemFeatures(osb_match.group(1)))

            if osp_match:
                feature6 = (self.getSystemFeatures(osp_match.group(1)))

            if sscv_match:
                feature7 = (self.getSSCVersion(sscv_match.group(1)))

        featurelist.append(feature1)
        featurelist.append(feature2)
        featurelist.append(feature3)
        featurelist.append(feature4)
        featurelist.append(feature5)
        featurelist.append(feature6)
        featurelist.append(feature7)

        # self.class_guess.append(self.getClass(cat_max))
        return featurelist

    def convertToWords(self, ssclog):
        log_as_string = []
        logstring = ""
        for executionn in ssclog.getExecutions():
            execution_as_string = []
            if executionn.hasProperties():
                for property in executionn.properties:
                    logstring = logstring + " " + property
            for logitemm in executionn.getlogitems():
                datetimestr = '[0-9\:]+\,\d+'
                dtsm = re.compile(datetimestr, re.IGNORECASE | re.DOTALL)
                ms = re.sub('[0-9\:]+\,\d+', " ", logitemm.logmessage)
                ms = re.sub('\d+\-\d+\-\d+', " ", ms)
                logstring = logitemm.methodname + " " + logitemm.loglevel + " " + ms
                if logitemm.hasExceptions():
                    for exception in logitemm.exception:
                        logstring = logstring + " " + exception.methodname + exception.description
                        if exception.hasStackTrace():
                            for st in exception.stacktrace:
                                logstring = logstring + " " + st.method
                execution_as_string.append(logstring)
            log_as_string.append(execution_as_string)
        return log_as_string

    def getClass(self, guess):
        if guess == "appserver":
            return 1
        elif guess == "database":
            return 2
        elif guess == "ldap":
            return 3
        elif guess == "quartz":
            return 4
        elif guess == "ssl":
            return 5
        else:
            return '0'

    def getFileNameFeature(self, filename):
        if filename in self.testfile_index.keys():
            return self.testfile_index[filename]
        else:
            count = self.testfile_index.__len__() + 1
            self.testfile_index[filename] = count
            return count

    def extractFeatures(self, parsedlog, targetCategory, testfilename):
        totalline = parsedlog.getLineCount()
        logtext = self.convertToWords(parsedlog)

        execcount = 0
        feature1 = 0

        featurelist = []

        for execution, execstring in zip(parsedlog.executions, logtext):
            execcount += 1
            if execstring.__len__() > 0:
                vectorizer = self.vectorizeWords(execstring)
                execfeatures = self.setCustomFeatures(vectorizer)

                lihe = 0  # Logitem has an exception
                warncount = 0  # Logitem has a WARN method
                errorcount = 0  # Logitem has a ERROR method
                endlinecount = float(execution.endline) / float(totalline)
                startlinecount = float(execution.startline) / float(totalline)

                for logitem in execution.getlogitems():
                    if logitem.hasExceptions(): lihe += 1
                    if logitem.loglevel == "WARN":
                        warncount += 1
                    elif logitem.loglevel == "ERROR":
                        errorcount += 1
                    continue

                guess = self.getClass(targetCategory)
                execfeatures.append(startlinecount)
                execfeatures.append(endlinecount)
                execfeatures.append(execution.getlogitemlength())
                execfeatures.append(lihe)
                execfeatures.append(warncount)
                execfeatures.append(errorcount)
                execfeatures.append(guess)
                featurelist.append(execfeatures)
            else:
                pass

        featuress = np.asarray(featurelist).astype(float)
        featurez = featuress.tolist()
        return featurez
