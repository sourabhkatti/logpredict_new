import re
from datetime import datetime
import hashlib

from classes import LogException, Execution, Stacktrace, Logitem


def __init__(self, filetoread):
    self.logmessages = filetoread


class totalWords():
    def __init__(self):
        self.wordrepo = []

    def addLog(self, logname, parsedwords):
        self.wordrepo[str(logname)] = parsedwords


def compareWords(log1, log2):
    count = 0
    z = {}
    probability = 0
    for word1 in log1.keys():
        for word2 in log2.keys():
            if (word1 == word2):
                if log2[word2] == 0:
                    probability = 0
                else:
                    probability = log1[word1] / log2[word2]
                count = count + 1
        z[word1] = probability
        count = 0
    return z


def parseUniqueWords(file):
    numlogs = +1
    uniquewords = {}
    with open(file, 'r') as g:
        line = g.read()
        wordcount = 0
        total = line.split()
        for word in total:
            wordcount = wordcount + 1
            if word not in uniquewords:
                uniquewords[word] = 1
            if word in uniquewords:
                uniquewords[word] = uniquewords[word] + 1
        normalized = normalizeWordCount(uniquewords, wordcount)
        return normalized


def normalizeWordCount(log, totalwords):
    normalizedVector = {}
    for key in log.keys():
        normalizedVector[key] = log[key] / totalwords
    return normalizedVector


def parselog(filetoread, ssclog):
    linenumber = 1
    exceptioncheck = False
    propertiesflag = False

    featureset = []

    e = Execution.Execution()
    e.startline = 1
    execnum = 1
    e.setID(execnum)
    li_id = 1
    exex_id = 0

    nullstring = '[\w\s\d\.\[\]\;\/\,\'\"\|\-\_\=\)\(\*\^\@]+'
    ns = re.compile(nullstring, re.IGNORECASE | re.DOTALL)

    li = Logitem.Logitem(datetime, 'test', 'test', 'test', 'test', 1)
    logdatetime = datetime

    with open(filetoread, 'r') as f:
        logmessages = f.readlines()
    f.close()

    for line in logmessages:
        # loglinedefaultpattern='([0-9\-]+\s[0-9\:\,]+)\s+\[(\w+)\]\s([\.\[\]\w]+)\s\-\s([a-z0-9\s\W\d]+)+'
        loglinedefaultpattern = '([0-9\-]+\s[0-9\:\,]+)\s+\[(\w+)\]\s([\.\w\S]+)\s\-\s([\w\S\ ]+)'
        logexceptionpattern = '([com|org|fortify\.\w\S]+)\:\s(.*)'
        stacktracepattern = '\s+at\s([a-z\w\.]+)\S+\(([a-zA-Z\_\<\>\.\:0-9\s]+)\)'
        causedbystring = 'Caused by\:\s([\w\.]+)\:\s(.*)'
        packetpattern = '(The\slast\spacket[\w\s\.]+)'

        catalina_logpattern = '(\w+\s\d+\,\s\d+\s[0-9\:]+\s\w+)\s([a-zA-Z\.\S0-9]+)\s(.*)'

        # setup regex matchers
        rg = re.compile(loglinedefaultpattern, re.IGNORECASE | re.DOTALL)
        lg = re.compile(logexceptionpattern, re.IGNORECASE | re.DOTALL)
        sg = re.compile(stacktracepattern, re.IGNORECASE | re.DOTALL)
        cg = re.compile(causedbystring, re.IGNORECASE | re.DOTALL)
        eg = re.compile(packetpattern, re.IGNORECASE | re.DOTALL)

        cslp = re.compile(catalina_logpattern, re.IGNORECASE | re.DOTALL)

        ns.match(line)

        # run match for regular log message
        m = rg.match(line)
        exceptionmatch = lg.match(line)

        nsg = ns.match(line)

        if nsg:

            catmatch = cslp.match(line)

            if line.__contains__("Fortify Runtime Properties"):
                propertiesflag = True
                continue

            elif propertiesflag and not m:
                e.properties.append(line)


            elif (exceptioncheck or exceptionmatch) and not propertiesflag:
                exceptionstring = ""
                causedbymatcher = cg.match(line)
                stacktracecheck = sg.match(line)
                packeterrorcheck = eg.match(line)

                if exceptionmatch:
                    try:
                        logexec.setID(li_id)
                        methodname = exceptionmatch.group(1)
                        logmessage = exceptionmatch.group(2)
                        logexec.methodname = methodname
                        logexec.description = logmessage
                    except:
                        catalina_string = '([\w]+)\:(.*)'
                        cm = re.compile(catalina_string, re.IGNORECASE | re.DOTALL)
                        cg = cm.match(line)
                        if cg:
                            li = Logitem.Logitem(datetime, cg.group(1), 'catalina', cg.group(2), execnum, linenumber)
                            e.addlogitem(li)
                elif stacktracecheck:
                    try:
                        string = stacktracecheck.group(2).split(':')
                        lineexec = string[2]
                        filename = string[1]
                    except:
                        lineexec = 0
                        filename = stacktracecheck.group(2)
                    stackmethod = stacktracecheck.group(1)
                    stacktrace = Stacktrace.Stacktrace(stackmethod, filename, lineexec, exex_id)
                    logexec.stacktrace.append(stacktrace)
                elif causedbymatcher:
                    logexec.setID(li_id)
                    exex_id = exex_id + 1
                    li.exception.append(logexec)
                    logmessage = causedbymatcher.group(2)
                    methodname = causedbymatcher.group(1)
                    logexec = LogException.Logexception(methodname, logmessage)
                    exex_id = exex_id + 1
                elif packeterrorcheck:
                    logmessage = packeterrorcheck.group(0)
                    logexec.description = logexec.description + logmessage
                elif m:
                    logexec.setID(li_id)
                    li.exception.append(logexec)
                    exex_id = exex_id + 1
                    li.setID(execnum)
                    e.addlogitem(li)
                    li_id = li_id + 1
                    exceptioncheck = False
                else:
                    linenumber = linenumber + 1
                    continue

                    # If it's a match, parse out relevant details
            if m:
                logdatetime = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if propertiesflag: propertiesflag = False
                loglevel = m.group(2)
                methodname = m.group(3)
                logmessage = m.group(4)
                li = Logitem.Logitem(logdatetime, loglevel, methodname, logmessage, execnum, linenumber)

                # check if it's the first line of the log, then set the execution's datetime to that date
                if (linenumber == 1):
                    e.starttime = logdatetime

                # check if there is a new execution. If so, set the starttime and startline of old execution, store it
                # in the logfile class, then start a new one
                if (methodname == "com.fortify.systemspec" and loglevel == 'WARN'):
                    linenumber = linenumber + 1
                    e.endtime = logdatetime
                    e.endline = linenumber
                    e.setID(execnum)
                    ssclog.addExecution(e)
                    e = Execution.Execution()
                    execnum = execnum + 1
                    e.setID(execnum)
                    e.startline = linenumber
                    e.starttime = logdatetime

                if (loglevel == 'ERROR'):
                    exceptioncheck = True
                    logexec = LogException.Logexception(methodname, logmessage)
                    logexec.setID(li_id)
                    exex_id = exex_id + 1

                else:
                    li.setID(li_id)
                    e.addlogitem(li)
                    li_id = li_id + 1

            if catmatch:
                logdatetime = datetime.strptime(catmatch.group(1), "%b %d, %Y %H:%M:%S %p")
                methodname = catmatch.group(2)
                logmessage = catmatch.group(3)
                catalina_li = Logitem.Logitem(logdatetime, 'catalina', methodname, logmessage, execnum, linenumber)
                e.addlogitem(catalina_li)


            else:
                e.properties.append(line)
            linenumber = linenumber + 1

        else:
            linenumber = linenumber + 1

    li.setID(li_id)
    li.linenumber = linenumber
    # e.addlogitem(li)
    e.endtime = logdatetime
    e.endline = linenumber
    ssclog.addExecution(e)
    ssclog.setLineCount(linenumber)

    return ssclog
