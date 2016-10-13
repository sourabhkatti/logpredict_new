from tkinter import *
import tkinter.filedialog as tkfd
import controlla
import os
import threading
import time
import queue
import numpy as np


# noinspection PyAttributeOutsideInit
class application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.queue = queue.Queue()
        self.qlength = 0

        self.ct = controlla.controlla()
        self.ttrainstop = threading.Event()

        # Setup container frames
        self.topframe = Frame(self.master, relief=FLAT, bg="#262626")
        self.bottomframe = Frame(self.master, relief=FLAT)
        self.rightframe = Frame(self.master, relief=FLAT)
        self.middleleftframe = Frame(self.master, relief=FLAT, bg="#000000")

        # Parsing buttons and text variables
        self.startparsertext = StringVar(value="Start Parser")
        self.startparser = Button(self.topframe, textvariable=self.startparsertext,
                                  command=self.startparsingfiles, bg="#E6E6F0",
                                  fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT)
        self.stopparsertext = StringVar(value="Stop Parser")
        self.stopparser = Button(self.topframe, textvariable=self.stopparsertext,
                                 command=self.stopparsingfiles, bg="#E6E6F0",
                                 fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT, state=DISABLED)

        # Training buttons and text variables
        self.starttrainertext = StringVar(value="Start Training")
        self.starttrainer = Button(self.middleleftframe, textvariable=self.starttrainertext,
                                   command=self.starttrainingmodels, bg="#E6E6F0",
                                   fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT)
        self.stoptrainertext = StringVar(value="Stop Training")
        self.stoptrainer = Button(self.middleleftframe, textvariable=self.stoptrainertext,
                                  command=self.stoptrainingmodel, bg="#E6E6F0",
                                  fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT, state=DISABLED)

        # Test trained models on logfile repository
        self.starttestertext = StringVar(value="Test Models")
        self.starttester = Button(self.middleleftframe, textvariable=self.starttestertext,
                                   command=self.startestingmodels, bg="#E6E6F0",
                                   fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT)
        self.stoptestertext = StringVar(value="Stop Testing")
        self.stoptester = Button(self.middleleftframe, textvariable=self.stoptestertext,
                                  command=self.stopestingmodels, bg="#E6E6F0",
                                  fg="#262626", font=("Ariel", 12, "bold"), relief=FLAT, state=DISABLED)

        # Test trained models on a new log file
        self.predictlogfile = Button(self.bottomframe, text="Predict New Log", font=("Ariel", 12, "bold"), command=self.open_predict_logfile,
                                     relief=FLAT, bg='#4286f4')
        self.quit = Button(self.bottomframe, text="Quit", fg="red", font=("Ariel", 12, "bold"), command=root.destroy,
                           relief=FLAT)

        # Console text field
        self.consoletext = Text(master=self.rightframe, height=3, width=50, relief=FLAT, state=DISABLED,
                                font=("Times New Roman", 12))
        self.consolescrollbar = Scrollbar(self.consoletext)
        self.consoletext['yscrollcommand'] = self.consolescrollbar.set

        # self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.rightframe.pack(side=RIGHT, anchor=E, expand=1, fill=BOTH)
        self.consoletext.pack(fill=BOTH, expand=1)
        self.consolescrollbar.pack(fill=BOTH, expand=1)

        self.topframe.pack(anchor=NW, fill=X, expand=1)
        self.startparser.pack(fill=X, expand=1)
        self.stopparser.pack(fill=X, expand=1)
        self.starttrainer.pack(fill=X, expand=1)
        self.stoptrainer.pack(fill=X, expand=1)
        self.starttester.pack(fill=X, expand=1)
        self.stoptester.pack(fill=X, expand=1)
        self.predictlogfile.pack(fill=X, expand=1, anchor=N)

        self.middleleftframe.pack(fill=X, expand=1, anchor=W)

        self.bottomframe.pack(side=BOTTOM, anchor=SW, expand=1, fill=X)
        self.quit.pack(fill=X, expand=1)

    def startparsingfiles(self):
        self.master.after(100, self.process_queue)
        self.stopparser['state'] = 'normal'
        message = "Starting parsing of all models\n"
        self.updateconsole(message)
        self.startparser['state'] = DISABLED

        self.queue.put("ParserStarted")

        ct1 = controlla.ThreadedTask_parser(self.queue, flag=self.ttrainstop)
        print("CT1 is ", ct1.is_alive())
        if not ct1.is_alive():
            ct1.start()

    def starttrainingmodels(self):
        self.master.after(100, self.process_queue)
        self.stoptrainer['state'] = 'normal'
        message = "Starting training of all models\n"
        self.updateconsole(message)
        self.starttrainer['state'] = DISABLED

        self.queue.put("TrainerStarted")

        ct1 = controlla.ThreadedTask_trainer(self.queue, flag=self.ttrainstop)
        print("CT1 is ", ct1.is_alive())
        if not ct1.is_alive():
            ct1.start()

    def startestingmodels(self):
        self.master.after(100, self.process_queue)
        self.stoptester['state'] = 'normal'
        message = "Starting testing of all models\n"
        self.updateconsole(message)
        self.starttester['state'] = DISABLED

        self.queue.put("TesterStarted")

        ct1 = controlla.ThreadedTask_tester(self.queue, flag=self.ttrainstop)
        print("CT1 is ", ct1.is_alive())
        if not ct1.is_alive():
            ct1.start()

    def updateconsole(self, text):
        self.consoletext.configure(state='normal')
        self.consoletext.insert(END, chars=text)
        self.consoletext.configure(state='disabled')

    def process_queue(self):
        try:
            if self.queue.queue.__len__() > self.qlength:
                self.qlength = self.queue.queue.__len__()

                msgs = self.queue.queue[-1] + '\n'
                self.updateconsole(msgs)
                self.master.after(1000, self.process_queue)
        except queue.Empty:
            self.master.after(100, self.process_queue)

    def stoptrainingmodel(self):
        # self.ttrainstop.set()
        message = 'Stopping training...\n'
        self.consoletext.insert(END, chars=message)
        self.master.after(100, self.process_queue)
        self.queue.put("trainstop")
        self.starttrainer['state'] = ACTIVE

    def stopestingmodels(self):
        # self.ttrainstop.set()
        message = 'Stopping testing...\n'
        self.consoletext.insert(END, chars=message)
        self.master.after(100, self.process_queue)
        self.queue.put("teststop")
        self.starttester['state'] = ACTIVE

    def stopparsingfiles(self):
        # self.ttrainstop.set()
        message = 'Stopping training...\n'
        self.consoletext.insert(END, chars=message)
        self.master.after(100, self.process_queue)
        self.queue.put("Stopped")
        self.startparser['state'] = ACTIVE

    def open_predict_logfile(self):
        testpath = tkfd.askopenfile(mode='r')
        message = "\n\nRunning model on %s\n" % testpath.name
        self.updateconsole(message)
        controlla.ThreadedTask_predictor_single_file(self.queue, path=testpath.name)
        preds = controlla.controlla.testnew_singlelog(self.ct, path=testpath.name, queue=self.queue)
        for model, prediction in preds.items():
            try:
                arg = np.argmax(prediction)
                if arg == 0:
                    message = model + " predicted this is a DEBUG SCA.LOG file\n"
                    self.updateconsole(message)
                elif arg == 1:
                    message = model + " predicted this is an SCA.LOG file\n"
                    self.updateconsole(message)
                elif arg == 2:
                    message = model + " predicted this is an SSC.LOG file\n"
                    self.updateconsole(message)
            except:
                message = model + " predicted a value of " + str(prediction) + "\n"
                self.updateconsole(message)


root = Tk()
root.configure(background="#262626")
app = application(master=root)

app.master.wm_title('Log Parser v1.0')
root.minsize(width=1000, height=1000)
tmain = threading.Thread(target=app.mainloop())
tmain.start()
