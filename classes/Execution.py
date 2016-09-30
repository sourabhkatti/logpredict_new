__author__ = 'Sourabh'
import datetime


class Execution:
    def __init__(self):
        self.id = 1
        self.logitems = []
        self.startline = int
        self.endline = int
        self.starttime = datetime
        self.endtime = datetime
        self.properties = []

    def addlogitem(self, logitem):
        self.logitems.append(logitem)

    def getProperties(self):
        return self.properties

    def getlogitems(self):
        return self.logitems

    def getlogitemlength(self):
        return self.logitems.__len__()

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id

    def hasProperties(self):
        if (self.properties.__len__()) > 0:
            return True
        else:
            return False
