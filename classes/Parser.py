import re
from datetime import datetime
import hashlib

from classes import LogException, Execution, Stacktrace, Logitem


def __init__(self, filetoread):
    self.logmessages = filetoread


class totalWords:
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

    with open(filetoread, 'r') as f:
        logmessages = f.readlines()
    f.close()

    linenumber = 0
    linecount = logmessages.__len__()

    while linenumber < linecount:
        line = logmessages[linenumber]

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
                            li = Logitem.Logitem(datetime, cg.group(1), '', cg.group(2), execnum, linenumber)
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


def parse(filetoread, logfile):
    exceptioncheck = False
    propertiesflag = False
    lm = False

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
    logmessage = ''

    # loglinedefaultpattern='([0-9\-]+\s[0-9\:\,]+)\s+\[(\w+)\]\s([\.\[\]\w]+)\s\-\s([a-z0-9\s\W\d]+)+'
    loglinedefaultpattern = '([0-9\-]+\s[0-9\:\,]+)\s+\[(\w+)\]\s([\.\w\S]+)\s\-\s([\w\S\ ]+)'
    scaloglinedefaultpattern = '\[([0-9\-]+\s[0-9\:\.0-9]+)\s([\w\.\S]+[Thread\-0-9\s]+)[Master\s]+([\w\s0-9]+)\]'
    logexceptionpattern = '([com|org|fortify\.\w\S]+)\:\s(.*)'
    stacktracepattern = '\s+at\s([a-z\w\.]+)\S+\(([a-zA-Z\_\<\>\.\:0-9\s]+)\)'
    causedbystring = 'Caused by\:\s([\w\.]+)\:\s(.*)'
    packetpattern = '(The\slast\spacket[\w\s\.]+)'

    catalina_logpattern = '(\w+\s\d+\,\s\d+\s[0-9\:]+\s\w+)\s([a-zA-Z\.\S0-9]+)\s(.*)'

    # setup regex matchers
    scarg = re.compile(scaloglinedefaultpattern, re.IGNORECASE | re.DOTALL)
    rg = re.compile(loglinedefaultpattern, re.IGNORECASE | re.DOTALL)
    lg = re.compile(logexceptionpattern, re.IGNORECASE | re.DOTALL)
    sg = re.compile(stacktracepattern, re.IGNORECASE | re.DOTALL)
    cg = re.compile(causedbystring, re.IGNORECASE | re.DOTALL)
    eg = re.compile(packetpattern, re.IGNORECASE | re.DOTALL)

    cslp = re.compile(catalina_logpattern, re.IGNORECASE | re.DOTALL)

    with open(filetoread, 'r') as f:
        logmessages = f.readlines()
    f.close()

    lineindex = 0
    linecount = logmessages.__len__()

    while lineindex < linecount:
        line = logmessages[lineindex]
        linenumber = lineindex + 1

        ns.match(line)

        # run match for regular log message
        m = rg.match(line)
        scm = scarg.match(line)
        exceptionmatch = lg.match(line)

        nsg = ns.match(line)

        if nsg:

            catmatch = cslp.match(line)

            if line.__contains__("Fortify Runtime Properties"):
                propertiesflag = True
                lineindex += 1
                continue

            elif line.__contains__("Runtime Information"):
                propertiesflag = True
                if linenumber > 1:
                    # Add execution to list
                    e.endtime = logdatetime
                    e.endline = linenumber
                    e.setID(execnum)
                    logfile.addExecution(e)

                    # Create a new execution
                    e = Execution.Execution()
                    execnum += 1
                    e.setID(execnum)
                    e.startline = linenumber
                    e.starttime = logdatetime
                else:
                    lineindex += 1
                    continue
            elif propertiesflag and not (m or scm):
                e.properties.append(line)
                lineindex += 1
                continue


            elif (exceptioncheck and exceptionmatch) and not (propertiesflag or lm):
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
                            li = Logitem.Logitem(datetime, cg.group(1), '', cg.group(2), execnum, linenumber)
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
                    lineindex += 1
                    continue

                    # If it's a match, parse out relevant details

            if lm and not (scm or propertiesflag):
                logmessage += line
                try:
                    test = logmessages[linenumber + 1]
                    if scarg.match(test):
                        li.logmessage = logmessage
                        li.setID(li_id)
                        e.addlogitem(li)
                        li_id += 1
                        lm = False
                except:
                    pass
            if m:
                logdatetime = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if propertiesflag:
                    propertiesflag = False
                loglevel = m.group(2)
                methodname = m.group(3)
                logmessage = m.group(4)
                li = Logitem.Logitem(logdatetime, loglevel, methodname, logmessage, execnum, linenumber)

                # check if it's the first line of the log, then set the execution's datetime to that date
                if lineindex == 0:
                    e.starttime = logdatetime

                # check if there is a new execution. If so, set the starttime and startline of old execution, store it
                # in the logfile class, then start a new one
                if methodname == "com.fortify.systemspec" and loglevel == 'WARN':
                    e.endtime = logdatetime
                    e.endline = linenumber
                    e.setID(execnum)
                    logfile.addExecution(e)
                    e = Execution.Execution()
                    execnum += 1
                    e.setID(execnum)
                    e.startline = linenumber
                    e.starttime = logdatetime

                if loglevel == 'ERROR':
                    exceptioncheck = True
                    logexec = LogException.Logexception(methodname, logmessage)
                    logexec.setID(li_id)
                    exex_id += 1

                else:
                    li.setID(li_id)
                    e.addlogitem(li)
                    li_id += 1

            if scm:
                try:
                    logdatetime = datetime.strptime(scm.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except:
                    logdatetime = datetime.strptime(scm.group(1), "%Y-%m-%d %H:%M:%S")
                if propertiesflag:
                    propertiesflag = False
                loglevel = scm.group(3)
                methodname = scm.group(2)
                logmessage = ''
                lm = True
                li = Logitem.Logitem(logdatetime, loglevel, methodname, logmessage, execnum, linenumber + 1)

                # check if it's the first line of the log, then set the execution's datetime to that date
                if lineindex == 0:
                    e.starttime = logdatetime

            if catmatch:
                logdatetime = datetime.strptime(catmatch.group(1), "%b %d, %Y %H:%M:%S %p")
                methodname = catmatch.group(2)
                logmessage = catmatch.group(3)
                catalina_li = Logitem.Logitem(logdatetime, 'catalina', methodname, logmessage, execnum, linenumber)
                e.addlogitem(catalina_li)
            lineindex += 1

        else:
            lineindex += 1

    li.setID(li_id)
    li.linenumber = linenumber
    # e.addlogitem(li)
    e.endtime = logdatetime
    e.endline = linenumber
    logfile.addExecution(e)
    logfile.setLineCount(linenumber)

    return logfile


def parseSCAlog(filetoread, scalog):
    linenumber = 1
    exceptioncheck = False
    propertiesflag = False
    lm = False

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

    linenumber = 0
    linecount = logmessages.__len__()

    # loglinedefaultpattern=        '\[([0-9\-]+\s[0-9\:\.0-9]+)\s([\w\.\S]+[Thread\-0-9\s]+)[Master\s]+[(\w\s0-9)]+\]'
    loglinedefaultpattern = '\[([0-9\-]+\s[0-9\:\.0-9]+)\s([\w\.\S]+[Thread\-0-9\s]+)[Master\s]+([\w\s0-9]+)\]'
    logexceptionpattern = '([com|org|fortify\.\w\S]+)\:\s([\w]+)'
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

    while linenumber < linecount:

        line = logmessages[linenumber]
        # print(line)
        ns.match(line)

        # run match for regular log message
        m = rg.match(line)
        exceptionmatch = lg.match(line)

        nsg = ns.match(line)

        if nsg:
            if line.__contains__("Runtime Information"):
                propertiesflag = True
                if linenumber > 1:
                    # Add execution to list
                    linenumber += 1
                    e.endtime = logdatetime
                    e.endline = linenumber - 1
                    e.setID(execnum)
                    scalog.addExecution(e)

                    # Create a new execution
                    e = Execution.Execution()
                    execnum += 1
                    e.setID(execnum)
                    e.startline = linenumber
                    e.starttime = logdatetime
                else:
                    linenumber += 1
                    continue

            elif propertiesflag and not m:
                e.properties.append(line)


            elif (exceptioncheck or exceptionmatch) and not (propertiesflag or lm):
                exceptionstring = ""
                print(line)
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
                        cm = re.compile(causedbystring, re.IGNORECASE | re.DOTALL)
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
                    exex_id += 1
                    li.exception.append(logexec)
                    logmessage = causedbymatcher.group(2)
                    methodname = causedbymatcher.group(1)
                    logexec = LogException.Logexception(methodname, logmessage)
                    exex_id += 1
                elif packeterrorcheck:
                    logmessage = packeterrorcheck.group(0)
                    logexec.description += logmessage
                elif m:
                    logexec.setID(li_id)
                    li.exception.append(logexec)
                    exex_id += 1
                    li.setID(execnum)
                    e.addlogitem(li)
                    li_id += 1
                    exceptioncheck = False
                else:
                    linenumber += 1
                    continue

                    # If it's a match, parse out relevant details

            if lm and not (m or propertiesflag):
                logmessage += line
                try:
                    test = logmessages[linenumber + 1]
                    if rg.match(test):
                        li.logmessage = logmessage
                        li.setID(li_id)
                        li.linenumber = linenumber + 1
                        e.addlogitem(li)
                        li_id += 1
                        lm = False
                except:
                    pass

            if m:
                try:
                    logdatetime = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except:
                    logdatetime = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if propertiesflag:
                    propertiesflag = False
                loglevel = m.group(3)
                methodname = m.group(2)
                logmessage = ''
                lm = True
                li = Logitem.Logitem(logdatetime, loglevel, methodname, logmessage, execnum, linenumber)

                # check if it's the first line of the log, then set the execution's datetime to that date
                if linenumber == 1:
                    e.starttime = logdatetime

                # check if there is a new execution. If so, set the starttime and startline of old execution, store it
                # in the logfile class, then start a new one


                if loglevel == 'ERROR':
                    exceptioncheck = True
                    logexec = LogException.Logexception(methodname, logmessage)
                    logexec.setID(li_id)
                    exex_id += 1

            linenumber += 1

        else:
            linenumber += 1

    li.setID(li_id)
    li.linenumber = linenumber + 1
    # e.addlogitem(li)
    e.endtime = logdatetime
    e.endline = linenumber
    scalog.addExecution(e)
    scalog.setLineCount(linenumber)

    return scalog
