__author__ = 'Sourabh'


class Logexception:
    def __init__(self, methodname, description):
        self.id=1
        self.methodname = methodname
        self.description = description
        self.stacktrace = []

    def hasStackTrace(self):
        if self.stacktrace.__len__()<1:
            return False
        else: return True

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id

