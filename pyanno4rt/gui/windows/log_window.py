# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'log_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_log_window(object):
    def setupUi(self, log_window):
        log_window.setObjectName("log_window")
        log_window.setWindowModality(QtCore.Qt.ApplicationModal)
        log_window.resize(550, 640)
        log_window.setMinimumSize(QtCore.QSize(550, 640))
        log_window.setMaximumSize(QtCore.QSize(560, 640))
        log_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.log_widget = QtWidgets.QWidget(log_window)
        self.log_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.log_widget.setObjectName("log_widget")
        self.close_pbutton = QtWidgets.QPushButton(self.log_widget)
        self.close_pbutton.setGeometry(QtCore.QRect(470, 600, 71, 31))
        self.close_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_pbutton.setObjectName("close_pbutton")
        self.log_tedit = QtWidgets.QTextEdit(self.log_widget)
        self.log_tedit.setGeometry(QtCore.QRect(10, 50, 531, 541))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.log_tedit.setFont(font)
        self.log_tedit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.log_tedit.setReadOnly(True)
        self.log_tedit.setObjectName("log_tedit")
        self.log_line = QtWidgets.QFrame(self.log_widget)
        self.log_line.setGeometry(QtCore.QRect(10, 40, 581, 2))
        self.log_line.setMinimumSize(QtCore.QSize(0, 2))
        self.log_line.setMaximumSize(QtCore.QSize(16777215, 2))
        self.log_line.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.log_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.log_line.setObjectName("log_line")
        self.log_label = QtWidgets.QLabel(self.log_widget)
        self.log_label.setGeometry(QtCore.QRect(10, 10, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.log_label.setFont(font)
        self.log_label.setStyleSheet("border: 0px solid;")
        self.log_label.setObjectName("log_label")
        self.close_pbutton.raise_()
        self.log_tedit.raise_()
        self.log_label.raise_()
        self.log_line.raise_()
        log_window.setCentralWidget(self.log_widget)

        self.retranslateUi(log_window)
        QtCore.QMetaObject.connectSlotsByName(log_window)

    def retranslateUi(self, log_window):
        _translate = QtCore.QCoreApplication.translate
        log_window.setWindowTitle(_translate("log_window", "Logging"))
        self.close_pbutton.setText(_translate("log_window", "Close"))
        self.log_tedit.setHtml(_translate("log_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p></body></html>"))
        self.log_label.setText(_translate("log_window", "Logs"))
