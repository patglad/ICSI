# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'icsi.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QIntValidator, QFont, QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QComboBox, \
    QMessageBox

import subprocess
import os


class Window(QMainWindow):

    def choose_weights(self):
        weights, _ = QFileDialog.getOpenFileName(self, 'Choose a weights file', '', 'Weights files | *.h5;')
        print(weights)
        return weights

    def choose_video(self):
        filePath, _ = QFileDialog.getOpenFileName(self, 'Choose a video file', '', 'Video files | *.avi;')
        print(filePath)
        return filePath

    def choose_dataset(self):
        dataset = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(dataset)
        return dataset

    def start_detection(self):
        filePath = self.choose_video()
        self.labelpath.setText('File: ' + filePath)
        url = QUrl.fromLocalFile(filePath)
        print(url.fileName())
        weights = self.choose_weights()

        if filePath and weights:
            command = r'python icsi.py splash --weights={} --video={}'.format(weights, filePath)
            print(command)
            subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             shell=True, cwd=r'D:\MASK-RCNN\samples\icsi')

    def start_training(self):
        if self.epochs.text() or self.steps.text():
            epochs_input = int(self.epochs.text())
            steps_input = int(self.steps.text())
            imGPU_input = int(self.imGPU.currentText())
            layers_input = self.layers.currentText()
            dataset = self.choose_dataset()
            weights = self.choose_weights()
            if dataset and weights:
                command = r'python icsi.py train --dataset={} --weights={} --epochs={} --steps={} --imGPU={} --layers={}' \
                    .format(dataset, weights, epochs_input, steps_input, imGPU_input, layers_input)
                print(command)
                subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 shell=True, cwd=r'D:\MASK-RCNN\samples\icsi')
        else:
            QMessageBox.warning(self, 'Warning', 'Set the parameters before training.', QMessageBox.Ok)

    def see_film(self):
        os.system('python videowindow.py')


    def setupUi(self, ICSIWindow):
        ICSIWindow.setObjectName("ICSIWindow")
        ICSIWindow.resize(285, 290) #bylo 318
        ICSIWindow.setMinimumSize(285, 290)
        ICSIWindow.setMaximumSize(285, 290)
        ICSIWindow.setStyleSheet("QMainWindow{\n"
                                 "background-image: url(:/nowyPrzedrostek/tlo.jpg);\n""}\n""")
        ICSIWindow.setWindowTitle("Stages of ICSI")
        #ICSIWindow.setWindowFlag(Qt.FramelessWindowHint) #--> hides the bar
        ICSIWindow.setWindowIcon(QIcon('ivf.png'))

        self.centralwidget = QWidget(ICSIWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Label displaying the file path of the chosen video
        self.labelpath = QLabel("File: ...", self.centralwidget)
        self.labelpath.setWordWrap(True)
        self.labelpath.setGeometry(20, 220, 800, 20)
        self.labelpath.setStyleSheet("color: black")


        ## Labels

        self.label = QtWidgets.QLabel("1. TRAINING", self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 5, 80, 21))
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-weight: bold;")

        self.label_2 = QtWidgets.QLabel("Epochs:", self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 111, 21))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel("Steps per epoch:", self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 60, 141, 21))
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel("Images per GPU:", self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 90, 131, 21))
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel("Layers:", self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 120, 80, 21))
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel("2. DETECTION", self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 190, 80, 21))
        self.label_6.setObjectName("label_6")
        self.label_6.setStyleSheet("font-weight: bold;")

        ## Input boxes

        self.epochs = QLineEdit(self.centralwidget)
        self.epochs.setObjectName("epochs")
        self.epochs.setGeometry(160, 30, 41, 21)
        self.epochs.setValidator(QIntValidator())

        self.steps = QLineEdit(self.centralwidget)
        self.steps.setObjectName("steps")
        self.steps.setGeometry(160, 60, 41, 21)
        self.steps.setValidator(QIntValidator())

        self.imGPU = QComboBox(self.centralwidget)
        self.imGPU.setObjectName("imGPU")
        self.imGPU.setGeometry(160, 90, 41, 21)
        self.imGPU.addItems(["1", "2"])

        self.layers = QComboBox(self.centralwidget)
        self.layers.setObjectName("layers")
        self.layers.setGeometry(160, 120, 69, 22)
        self.layers.addItems(["heads", "3+", "4+", "5+", "all"])

        ## Button to train

        self.train = QPushButton("Start training", self.centralwidget)
        self.train.clicked.connect(self.start_training)
        self.train.setGeometry(20, 155, 120, 21)
        self.train.setFont(QFont('Arial', 10))
        self.train.setStyleSheet("background-color:white;\n"
                                 "color: black;\n"
                                 "font-weight: bold;"
                                 "")

        ## Button to detect

        self.detection = QPushButton("Run detection", self.centralwidget)
        self.detection.clicked.connect(self.start_detection)
        self.detection.setGeometry(20, 260, 120, 21)
        self.detection.setFont(QFont('Arial', 10))
        self.detection.setStyleSheet("background-color:white;\n"
                                     "color: black;\n"
                                     "font-weight: bold;"
                                     "")

        ## Button to diplay film

        self.film = QPushButton("Visualisation", self.centralwidget)
        self.film.clicked.connect(self.see_film)
        self.film.setGeometry(145, 260, 120, 21)
        self.film.setFont(QFont('Arial', 10))
        self.film.setStyleSheet("background-color:white;\n"
                                     "color: black;\n"
                                    "font-weight: bold;"
                                     "")


        ICSIWindow.setCentralWidget(self.centralwidget)
        #self.menubar = QMenuBar(ICSIWindow)
        #self.menubar.setGeometry(0, 0, 960, 21)
        #self.menubar.setObjectName("menubar")
        #ICSIWindow.setMenuBar(self.menubar)
        #self.statusbar = QStatusBar(ICSIWindow)
        #self.statusbar.setObjectName("statusbar")
        #ICSIWindow.setStatusBar(self.statusbar)


#import images.tlo

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ICSIWindow = QtWidgets.QMainWindow()
    ui = Window()
    ui.setupUi(ICSIWindow)
    ICSIWindow.show()
    sys.exit(app.exec_())
