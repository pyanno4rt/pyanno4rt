# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'text_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_text_window(object):
    def setupUi(self, text_window):
        text_window.setObjectName("text_window")
        text_window.setWindowModality(QtCore.Qt.ApplicationModal)
        text_window.resize(400, 250)
        text_window.setMinimumSize(QtCore.QSize(400, 250))
        text_window.setMaximumSize(QtCore.QSize(400, 250))
        text_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.body_widget = QtWidgets.QWidget(text_window)
        self.body_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.body_widget.setObjectName("body_widget")
        self.close_text_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.close_text_pbutton.setGeometry(QtCore.QRect(320, 210, 71, 31))
        self.close_text_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_text_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_text_pbutton.setObjectName("close_text_pbutton")
        self.text_tedit = QtWidgets.QTextEdit(self.body_widget)
        self.text_tedit.setGeometry(QtCore.QRect(10, 10, 381, 191))
        self.text_tedit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.text_tedit.setReadOnly(True)
        self.text_tedit.setObjectName("text_tedit")
        text_window.setCentralWidget(self.body_widget)

        self.retranslateUi(text_window)
        QtCore.QMetaObject.connectSlotsByName(text_window)

    def retranslateUi(self, text_window):
        _translate = QtCore.QCoreApplication.translate
        text_window.setWindowTitle(_translate("text_window", "Text Viewer"))
        self.close_text_pbutton.setText(_translate("text_window", "Close"))
        self.text_tedit.setHtml(_translate("text_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
