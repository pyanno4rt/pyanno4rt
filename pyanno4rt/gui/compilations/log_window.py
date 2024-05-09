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
        log_window.resize(720, 640)
        log_window.setMinimumSize(QtCore.QSize(720, 640))
        log_window.setMaximumSize(QtCore.QSize(720, 640))
        log_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.log_widget = QtWidgets.QWidget(log_window)
        self.log_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.log_widget.setObjectName("log_widget")
        self.close_log_pbutton = QtWidgets.QPushButton(self.log_widget)
        self.close_log_pbutton.setGeometry(QtCore.QRect(640, 600, 71, 31))
        self.close_log_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_log_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_log_pbutton.setObjectName("close_log_pbutton")
        self.log_tedit = QtWidgets.QTextEdit(self.log_widget)
        self.log_tedit.setGeometry(QtCore.QRect(10, 10, 701, 581))
        self.log_tedit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.log_tedit.setReadOnly(True)
        self.log_tedit.setObjectName("log_tedit")
        log_window.setCentralWidget(self.log_widget)

        self.retranslateUi(log_window)
        QtCore.QMetaObject.connectSlotsByName(log_window)

    def retranslateUi(self, log_window):
        _translate = QtCore.QCoreApplication.translate
        log_window.setWindowTitle(_translate("log_window", "Logging"))
        self.close_log_pbutton.setText(_translate("log_window", "Close"))
        self.log_tedit.setHtml(_translate("log_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
