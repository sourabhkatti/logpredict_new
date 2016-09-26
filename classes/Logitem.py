__author__ = 'Sourabh'
from classes import LogException

class Logitem:
    def __init__(self, datetimestamp, loglevel, methodname, logmessage, execution_id, linenumber):
        self.datetimestamp = datetimestamp
        self.loglevel = loglevel
        self.methodname = methodname
        self.logmessage = logmessage
        self.exception = []
        self.hash = hash
        self.execution_id = execution_id
        self.linenumber = linenumber
        self.id=1

    def hasExceptions(self):
        if self.exception.__len__()<1:
            return False
        else: return True

    def getExceptions(self):
        return self.exception

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id

