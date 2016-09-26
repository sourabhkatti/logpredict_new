__author__ = 'Sourabh'
from classes import Parser

class Logfile:
    def __init__(self):
        self.scantype = type
        self.executions = []
        self.totallines = 0

    def addExecution(self, Execution):
        self.executions.append(Execution)

    def getExecutions(self):
        return self.executions

    def getlogitemsize(self):
        count = 0
        for execution in self.executions:
            count+=execution.getlogitemlength()
        return count

    def setLineCount(self, linecount):
        self.totallines = linecount

    def getLineCount(self):
        return self.totallines

    def parseFile(self, file):
        ssclog = Logfile()
        ssclog.scantype='ssc'
        return Parser.parselog(file, ssclog)