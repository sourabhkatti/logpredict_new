__author__ = 'Sourabh'


class Stacktrace:
    def __init__(self, method, filename, linenumber, id):
        self.id = id
        self.method = method
        self.filename = filename
        self.linenumber = linenumber

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id
