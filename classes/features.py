__author__ = 'Sourabh'

from classes import Logfile
from classes import *
from classes import Logitem
from classes.Logfile import *
from classes import Execution
from classes.Execution import *
from classes.features import *
from classes.LogException import *
from classes.Stacktrace import *
import operator
import re
import string

class classifier():

    master_feature_list={}


    def __init__(self):
        master_list={}

    def addCat (self, category, keywords):
        if category in self.master_feature_list.keys():
            if keywords.__class__ == bool:
                self.master_feature_list[category]=keywords
                return
            for keyword,count in keywords.items():
                if keyword in self.master_feature_list[category].keys():
                    self.master_feature_list[category][keyword]+=count
                else:
                    self.master_feature_list[category][keyword]=count
        else:
            self.master_feature_list[category]=keywords

    def fprob(self,keyword,category):
        if not (category in self.classify.master_feature_list.keys()or keyword in self.classify.master_feature_list[category].keys()): return 0.0
        # The total number of times this feature appeared in this
        # category divided by the total number of items in this category
        keywordcount =   float(self.master_feature_list[category][keyword])
        totalcount =    float(self.master_feature_list[category]['total'])
        test = float(keywordcount/totalcount)
        return test


    def convertToProbability(self, dict_count):
        sorted_dict={}
        if sum(dict_count.values()) == 0:
            return None
        for category in dict_count:
            if category == 'total': continue
            probability=float(dict_count[category])/float(sum(dict_count.values()))
            if probability > 0: sorted_dict[category]=probability
        return sorted_dict

    def weightedProbabability(self,keyword,category,weight=1.0,ap=0.5):
        # Calculate current probability
        basicprob=self.fprob(keyword,category)
        # Count the number of times this feature has appeared in
        # all categories
        totals=0
        for categoryiter, keyworditer in self.master_feature_list.iteritems():
            if not (category in self.classify.master_feature_list.keys()or keyword in self.classify.master_feature_list[category].keys()): return 0.0
            else:
                totals+=self.master_feature_list[category][keyword]
        # Calculate the weighted average
        bp=((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp

    def rankDict(self, dict_to_rank):
        if dict_to_rank==None: return None
        ranked_dict = sorted(dict_to_rank.items(), key=operator.itemgetter(1))
        return  dict(ranked_dict)

    def rankKeywords(self):
        ranked_category = {}
        for category in self.master_feature_list.keys():
            if not self.convertToProbability(self.master_feature_list[category])== None:
                ranked_list = self.convertToProbability(self.master_feature_list[category])
                sorted_x = sorted(ranked_list.items(), key=operator.itemgetter(1))
                ranked_category[category]=sorted_x
        return ranked_category

    def rankCategories(self):
        ranked_cat = {}
        for category in self.master_feature_list.keys():
            if not (self.master_feature_list[category]['total'])== 0:
                ranked_cat[category]=self.master_feature_list[category]['total']
        ranked_list = self.convertToProbability(ranked_cat)
        sorted_x = sorted(ranked_list.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_x


    def printRankedFeatures(self):
        ranked_list = self.rankExecutions()
        for key in ranked_list:
            print("\n")
            print (key)
            print (ranked_list[key])







class naivebayes():

    classify = classifier()
    execution_weights = dict

    def setExecutionFeature(self, logfileweight):
        self.execution_weights = logfileweight

    def setClassifier(self, classifiertrue):
        self.classify = classifiertrue

    def weightedProbabability(self,keyword,category,weight=1.0,ap=1.0):
        # Calculate current probability
        basicprob=self.fprob(keyword,category)
        # Count the number of times this feature has appeared in
        # all categories
        totals=0
        for categoryiter, keyworditer in self.classify.master_feature_list.items():
            if not (category in self.classify.master_feature_list.keys()or keyword in self.classify.master_feature_list[category].keys()): return 0.0
            else:
                totals+=self.classify.master_feature_list[category][keyword]
                # Calculate the weighted average
        bp=((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp

    def cprob(self, category):
        totals=0
        if not (self.classify.master_feature_list.has_key(category)): return 0.0
        for category, keywords in self.classify.master_feature_list.iteritems:
            totals+=keywords['total']
        return (self.classify.master_feature_list[category]['total']/totals)

    def fprob(self,keyword,category):
        if not (category in self.classify.master_feature_list.keys()or keyword in self.classify.master_feature_list[category].keys()): return 0.0
        # The total number of times this feature appeared in this
        # category divided by the total number of items in this category
        keywordcount =   float(self.classify.master_feature_list[category][keyword])
        totalcount =    float(self.classify.master_feature_list[category]['total'])
        test = float(keywordcount/totalcount)
        return test

    def printFeatures(self):
        for category in self.master_feature_list:
            print ("\n"+category)
            print (self.master_feature_list[category])
        listtt = self.sortDictionary()
        print (listtt)

    def docprob(self,item,category):

        p=1
        for feature in self.classify.master_feature_list[category]: p*=self.weightedProbabability(feature,category)
        return p

    def prob(self,item,category):
        execution_prob={}
        if not (category in self.classify.master_feature_list.keys()or item in self.classify.master_feature_list[category].keys()): return 0.0

        for execution,category_count in self.execution_weights.items():

            if category_count[category]==0: execution_prob[execution]=0.0
            else:
                catprob=float(self.classify.master_feature_list[category]['total'])/float(category_count[category])
                docprob=float(self.docprob(item,category))
                execution_prob[execution]= docprob*catprob
        return execution_prob








class Features:

    categories = {}
    subcategories = {}
    classify = classifier()

    logfileWeights = {}
    executionWeights = {}
    logitemWeights = {}
    categoryWeights = {}
    keywordWeights = {}
    totalWeights = {}
    keywordProbWeight = {}
    values_to_write = ''
    log_to_write = {}
    keyword_master_list = {}
    keyword_columns = ''

    i=0

    uniquewords = {}
    lineWeights = list()

    categories['mysql']=('mysql', 'MySQL', 'MYSQL','MysqlParameterMetadata','com.mysql.jdbc.SQLError.createSQLException')
    categories['oracle']=('oracle','ORACLE', 'Oracle')
    categories['memory']=("memory", "GC", "resources", "DEBUG", "MaxPermSize")

    categories['connection_problem']=('packet', 'packet sent', 'received any packets', 'FortifyException', 'ConnectException', 'Connection', 'timed', 'connect','java.net.DualStackPlainSocketImpl.connect0(Native Method)', 'Connection timed out', 'java.net.ConnectException: Connection timed out: connect', 'accessible',
                                      'not accessible', 'not down', 'overloaded', 'Connection refused', 'Connection refused: connect',  )

    categories['rulepacks']=('RulepackBLLImpl', 'com.fortify.manager.BLL.impl.RulepackBLLImpl', 'importinstalledrulepacks', 'Exception importinstalledrulepacks', 'FortifyException', 'security content', 'error', 'downloading', 'There was an error downloading security content')

    categories['ssl']=('sun.security.validator.ValidatorException', 'ValidatorException', 'PKIX', 'BugTrackerException', 'com.fortify.pub.bugtracker.support.BugTrackerException',
                       'SSLHandshakeException', 'PKIX path building failed', 'SunCertPathBuilderException', 'certification', 'valid certification path', 'unable to find valid certification path to requested target',
                       'sun.security.provider.certpath.SunCertPathBuilderException', 'PKIXValidator')

    categories['database_connection']=('hibernate','DB2', 'mysql', 'SQL Error', 'SQLException' 'com.fortify.manager.service.scheduler.SchedulerManagerImpl', 'jdbc', 'driver', 'SqlExceptionHelper', 'Communications link failure', 'org.hibernate.engine.jdbc.spi.SqlExceptionHelper', 'Communications link failure', 'Hibernate Session',
                                       'transaction', 'could not extract ResultSet', 'JDBCConnectionException', 'Could not open connection', 'SQLState', 'SQL Error:')

    categories['runtime']=('RuntimeControllerConnectionConfiguration', 'Initializing runtime', 'event handlers', 'com.fortify.manager.service.runtime.RuntimeControllerConnectionConfiguration', 'Initializing runtime event handlers', '10234', 'com.fortify.manager.service.runtime.RuntimeControllerConnectionConfiguration',
                           )

    categories['ssl_connection']=('certification', 'unable to find valid certification path to requested target', 'sun.security.provider.certpath.SunCertPathBuilderException', 'sun.security.provider.certpath.SunCertPathBuilderException',
                                  'sun.security.validator.ValidatorException', 'org.apache.xmlrpc.XmlRpcException',  'javax.net.ssl.SSLHandshakeException', 'java.security.cert.CertificateException', 'No subject alternative names present')

    categories['jira']=('JIRA', 'jira')

    categories['bugzilla'] = ('Bugzilla', 'com.fortify.pub.bugtracker.support.BugTrackerAuthenticationException','com.fortify.pub.bugtracker.support.BugTrackerAuthenticationException', 'Bugzilla4BugTrackerPlugin',
                              'BugTrackerBLLImpl')


    categories['authorization']=('ScanProcessAuthorizationException', 'authorization', 'REQUIRE_AUTHORIZATION')

    categories['upload_fail']=('SessionTimeoutFilter', 'ResultFileUpload', 'uploadAnalysisResult','LoadFailedException', 'Upload artifact', 'Upload artifact failed for the following reason', 'com.fortify.manager.service.parser.checker.ScanProcessAuthorizationException', 'Processing Messages:')

    categories['port_issue']=('port', '10234', 'com.fortify.runtime.config.UserError', 'port 10234 already in use')

    categories['authentication']=('com.fortify.manager.service.parser.checker.ScanProcessAuthorizationException', 'Authentication', 'AuthenticationSuccessEvent', 'Authentication event AuthenticationSuccessEvent: admin')

    categories['login_error']=('Could not login', 'Could not login to Bugzilla server at' )

    categories['train']=('FATAL', 'Sourabh')

    categories['report_generation']=('org.eclipse.birt.report.engine.layout.html', 'org.eclipse.birt.data.engine.odaconsumer', 'parameter', 'birt', 'ReportBLLImpl', 'report', 'report generation', 'Could not find or load main class Files', 'jobCallback')

    categories['qrtzjob']=('Trigger', 'PV$','scheduler' )

    categories['fvdl']=('The fvdl file in this project is version', '1.12', 'fvdl','checkFVDLVersion', )


    def resetFeatures(self):
        for category in self.categories:
            self.executionWeights[category]=0
            self.totalWeights[category]=0
            self.categoryWeights[category]=0

    def convertToWords(self, ssclog):
        execution_words={}
        logitem_words={}
        for executionn in ssclog.executions:
            assert isinstance(executionn, Execution)
            execution_as_string=""
            for logitemm in executionn.getlogitems():
                assert isinstance(logitemm, Logitem.Logitem)
                logstring = logitemm.methodname + " " + logitemm.loglevel + " " + logitemm.logmessage
                if logitemm.hasExceptions():
                    for exception in logitemm.exception:
                        assert isinstance(exception, Logexception)
                        logstring=logstring + " " + exception.methodname + exception.description

                        if exception.hasStackTrace():
                            for st in exception.stacktrace:
                                assert isinstance(st, Stacktrace)
                                logstring=logstring+ " " + st.method
                execution_as_string=execution_as_string+ " " + logstring
            execution_words[executionn]=execution_as_string

        return execution_words

    def getClassifier(self):
        return self.classify

    def extractFeatures(self, ssclog):
        self.resetFeatures()
        execution_as_string = ""
        for executionn in ssclog.executions:
            assert isinstance(executionn, Execution)
            self.resetFeatures()
            for logitemm in executionn.getlogitems():
                assert isinstance(logitemm, Logitem.Logitem)
                logstring = logitemm.methodname + " " + logitemm.loglevel + " " + logitemm.logmessage
                if logitemm.hasExceptions():
                    for exception in logitemm.exception:
                        assert isinstance(exception, Logexception)
                        logstring=logstring + " " + exception.methodname + exception.description

                        if exception.hasStackTrace():
                            for st in exception.stacktrace:
                                assert isinstance(st, Stacktrace)
                                logstring=logstring+ " " + st.method
                                execution_as_string=execution_as_string+logstring

                weightsum=0




                #Loop through all categories for each log item
                for category in self.categories.keys():
                    categorycount=0
                    for keyword in self.categories[category]:
                        keywordcount=logstring.count(keyword)
                        self.keywordWeights[keyword]=0
                        if (keywordcount>0)  :
                            self.keywordWeights[keyword]=keywordcount
                            categorycount=categorycount+keywordcount
                        self.addKeywordToFile(keyword, keywordcount)
                    self.keywordWeights['total']=categorycount

                    self.classify.addCat(category, self.keywordWeights)
                    self.categoryWeights[category]=categorycount
                    self.executionWeights[category]=self.executionWeights[category]+categorycount
                    self.addToKeywordMaster()
                    self.keywordWeights={}

                self.addLineToFile()
                    #print (executionWeights)
            self.logfileWeights[executionn]=self.executionWeights
            self.executionWeights={}
            storage.writeLines(self.lineWeights,self.keyword_columns )
        return self.logfileWeights

    def addKeywordToFile(self, keyword, value):
        self.keyword_columns = self.keyword_columns + str(keyword) + "\t"
        self.values_to_write = self.values_to_write + str(value) + "\t"

    def addLineToFile(self):
        self.lineWeights.append(self.values_to_write)
        self.values_to_write = ''

    def printFeatures(self):
        for execprint in self.logfileWeights.values():
            print(execprint)


    def keywordCount(self,keyword,category):
        if category in self.categories[category] and keyword in self.keywordWeights[keyword]:
            return float(self.keywordWeights[keyword])
        return 0.0

    def addToKeywordMaster(self):
        for keyword in self.keywordWeights.keys():
            if keyword in self.keyword_master_list.keys():
                self.keyword_master_list[keyword]=self.keyword_master_list[keyword]+self.keywordWeights[keyword]
            else:
                self.keyword_master_list[keyword]=self.keywordWeights[keyword]


    def normalizeWordCount(self, log, totalwords):
        normalizedVector = {}
        for key in log.keys():
            normalizedVector[key]=log[key]/totalwords
        return normalizedVector

    def parseUniqueWords(self, line):
        wordcount=0
        total = line.split()
        for word in total:
            wordcount=wordcount+1
            if word not in self.uniquewords:
                self.uniquewords[word]=1
            if word in self.uniquewords:
                self.uniquewords[word]=self.uniquewords[word]+1
        normalized = self.normalizeWordCount(self.uniquewords, wordcount)
        return normalized


    def keywordProb(self):
        dict=self.keywordWeights
        if sum(self.keywordWeights.values())==0: return dict
        for keyword in self.keywordWeights:
            dict[keyword]=self.keywordWeights[keyword]/(sum(self.keywordWeights.values()))
        return dict


    def getwords(self, executionwords):
        for doc in executionwords.values():

            splitter=re.compile('\\W*')
            # Split the words by non-alpha characters
            words=[s.lower( ) for s in splitter.split(doc)
                   if len(s)>2 and len(s)<20]
            # Return the unique set of words only
            return dict([(w,1) for w in words])

    def getExecProb(self):
        cat_probability={}
        test = {}
        for execution,catcount in self.logfileWeights.items():
            test = self.classify.convertToProbability(catcount)
            cat_probability[execution] = self.classify.rankDict(test)
        return cat_probability





    def getCategories(self):
        return self.categories