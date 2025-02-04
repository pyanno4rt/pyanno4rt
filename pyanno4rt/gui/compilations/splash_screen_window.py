# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'splash_screen_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_splash_window(object):
    def setupUi(self, splash_window):
        splash_window.setObjectName("splash_window")
        splash_window.setWindowModality(QtCore.Qt.ApplicationModal)
        splash_window.resize(410, 320)
        splash_window.setWindowOpacity(1.0)
        self.splash_widget = QtWidgets.QWidget(splash_window)
        self.splash_widget.setObjectName("splash_widget")
        self.image_frame = QtWidgets.QFrame(self.splash_widget)
        self.image_frame.setGeometry(QtCore.QRect(0, -90, 410, 421))
        self.image_frame.setStyleSheet("image: url(:/Logo/logo_white_square.png);\n"
"background-color: transparent;\n"
"border: 0px solid;\n"
"border-color: rgb(0, 0, 0);\n"
"border-radius: 15px;")
        self.image_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.image_frame.setObjectName("image_frame")
        self.text_frame = QtWidgets.QFrame(self.splash_widget)
        self.text_frame.setGeometry(QtCore.QRect(1, 202, 411, 140))
        self.text_frame.setStyleSheet("border: 0px solid; background-color: transparent")
        self.text_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.text_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.text_frame.setObjectName("text_frame")
        self.version_label = QtWidgets.QLabel(self.text_frame)
        self.version_label.setGeometry(QtCore.QRect(0, 20, 411, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.version_label.setFont(font)
        self.version_label.setStyleSheet("background-color: transparent;\n"
"color: rgb(254, 144, 41);")
        self.version_label.setAlignment(QtCore.Qt.AlignCenter)
        self.version_label.setObjectName("version_label")
        self.progressBar = QtWidgets.QProgressBar(self.text_frame)
        self.progressBar.setGeometry(QtCore.QRect(16, 80, 379, 30))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"        background-color: transparent;\n"
"}\n"
"QProgressBar::chunk{\n"
"       background-color: qlineargradient(x0: 0, x2: 1,  stop: 0 #4664aa, stop: 0.11765 #009682, stop: 0.2353 #8cb63c, stop: 0.3529 #a7822e, stop: 0.4706 #df9b1b, stop: 0.588 #a22223);\n"
"}")
        self.progressBar.setMinimum(0)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.init_label = QtWidgets.QLabel(self.text_frame)
        self.init_label.setGeometry(QtCore.QRect(-1, 50, 411, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.init_label.setFont(font)
        self.init_label.setStyleSheet("background-color: transparent;\n"
"color: rgb(211, 215, 207);")
        self.init_label.setAlignment(QtCore.Qt.AlignCenter)
        self.init_label.setObjectName("init_label")
        splash_window.setCentralWidget(self.splash_widget)

        self.retranslateUi(splash_window)
        QtCore.QMetaObject.connectSlotsByName(splash_window)

    def retranslateUi(self, splash_window):
        _translate = QtCore.QCoreApplication.translate
        splash_window.setWindowTitle(_translate("splash_window", "pyanno4rt"))
        self.init_label.setText(_translate("splash_window", "Launching Graphical User Interface ..."))
