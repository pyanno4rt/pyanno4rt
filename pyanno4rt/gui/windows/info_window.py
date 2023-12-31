# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'info_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_info_window(object):
    def setupUi(self, info_window):
        info_window.setObjectName("info_window")
        info_window.setWindowModality(QtCore.Qt.ApplicationModal)
        info_window.resize(730, 540)
        info_window.setMinimumSize(QtCore.QSize(730, 540))
        info_window.setMaximumSize(QtCore.QSize(730, 540))
        info_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.info_widget = QtWidgets.QWidget(info_window)
        self.info_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.info_widget.setObjectName("info_widget")
        self.about_line = QtWidgets.QFrame(self.info_widget)
        self.about_line.setGeometry(QtCore.QRect(20, 50, 691, 2))
        self.about_line.setMinimumSize(QtCore.QSize(0, 2))
        self.about_line.setMaximumSize(QtCore.QSize(16777215, 2))
        self.about_line.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.about_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.about_line.setObjectName("about_line")
        self.about_label = QtWidgets.QLabel(self.info_widget)
        self.about_label.setGeometry(QtCore.QRect(20, 20, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.about_label.setFont(font)
        self.about_label.setStyleSheet("border: 0px solid;")
        self.about_label.setObjectName("about_label")
        self.about_tbrowser = QtWidgets.QTextBrowser(self.info_widget)
        self.about_tbrowser.setGeometry(QtCore.QRect(20, 60, 691, 301))
        self.about_tbrowser.setStyleSheet("border: 0px;")
        self.about_tbrowser.setReadOnly(True)
        self.about_tbrowser.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard|QtCore.Qt.LinksAccessibleByMouse)
        self.about_tbrowser.setOpenExternalLinks(True)
        self.about_tbrowser.setObjectName("about_tbrowser")
        self.thanks_line = QtWidgets.QFrame(self.info_widget)
        self.thanks_line.setGeometry(QtCore.QRect(20, 400, 691, 2))
        self.thanks_line.setMinimumSize(QtCore.QSize(0, 2))
        self.thanks_line.setMaximumSize(QtCore.QSize(16777215, 2))
        self.thanks_line.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.thanks_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.thanks_line.setObjectName("thanks_line")
        self.thanks_label = QtWidgets.QLabel(self.info_widget)
        self.thanks_label.setGeometry(QtCore.QRect(20, 370, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.thanks_label.setFont(font)
        self.thanks_label.setStyleSheet("border: 0px solid;")
        self.thanks_label.setObjectName("thanks_label")
        self.thanks_tbrowser = QtWidgets.QTextBrowser(self.info_widget)
        self.thanks_tbrowser.setGeometry(QtCore.QRect(20, 410, 691, 61))
        self.thanks_tbrowser.setStyleSheet("border: 0px;")
        self.thanks_tbrowser.setReadOnly(True)
        self.thanks_tbrowser.setObjectName("thanks_tbrowser")
        self.close_pbutton = QtWidgets.QPushButton(self.info_widget)
        self.close_pbutton.setGeometry(QtCore.QRect(640, 500, 71, 31))
        self.close_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_pbutton.setObjectName("close_pbutton")
        self.about_label.raise_()
        self.about_line.raise_()
        self.about_tbrowser.raise_()
        self.thanks_label.raise_()
        self.thanks_line.raise_()
        self.thanks_tbrowser.raise_()
        self.close_pbutton.raise_()
        info_window.setCentralWidget(self.info_widget)

        self.retranslateUi(info_window)
        QtCore.QMetaObject.connectSlotsByName(info_window)

    def retranslateUi(self, info_window):
        _translate = QtCore.QCoreApplication.translate
        info_window.setWindowTitle(_translate("info_window", "Info"))
        self.about_label.setText(_translate("info_window", "About pyanno4rt"))
        self.about_tbrowser.setHtml(_translate("info_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">pyanno4rt</span> is an open-source Python package for conventional and outcome prediction model-based inverse photon and proton treament plan optimization, including radiobiological and machine learning models for tumor control probability and normal tissue complication probability.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This package has been started by Tim Ortkamp as part of the doctoral project &quot;Inverse Radiotherapy Treatment Planning using Machine Learning Outcome Prediction Models&quot;, which evolved from a joint research effort by the <span style=\" font-style:italic;\">Karlsruhe Institute of Technology</span> (KIT), the <span style=\" font-style:italic;\">German Cancer Research Center</span> (DKFZ), and the <span style=\" font-style:italic;\">Helmholtz Information and Data Science School for Health</span> (HIDSS4Health).</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">More information about the package can be found at <a href=\"http://github.com/pyanno4rt/pyanno4rt\"><span style=\" text-decoration: underline; color:#0000ff;\">http://github.com/pyanno4rt/pyanno4rt</span></a>, where the open-source code is hosted as well.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">pyanno4rt</span> is under continuous development. Questions or suggestions for improvement can be sent via e-mail to <a href=\"mailto:tim.ortkamp@kit.edu?subject=pyanno4rt\"><span style=\" text-decoration: underline; color:#0000ff;\">tim.ortkamp@kit.edu </span></a>or via <a href=\"https://github.com/pyanno4rt/pyanno4rt/discussions\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/pyanno4rt/pyanno4rt/discussions</span></a>.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Distributed under the terms of the GNU GPLv3 license.</p></body></html>"))
        self.thanks_label.setText(_translate("info_window", "Special thanks"))
        self.thanks_tbrowser.setHtml(_translate("info_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" text-decoration: underline;\">Prof. Dr. Martin Frank</span> (KIT), <span style=\" text-decoration: underline;\">Prof. Dr. Oliver Jäkel </span>and <span style=\" text-decoration: underline;\">Dr. Niklas Wahl</span> (both DKFZ) for the project supervision. <span style=\" text-decoration: underline;\">Moritz Müller</span> for the student assistance in the project. <span style=\" text-decoration: underline;\">My colleagues at KIT, DKFZ and HIDSS4Health</span> for valuable discussions and feedbacks. Many thanks for all your contributions!</p></body></html>"))
        self.close_pbutton.setToolTip(_translate("info_window", "Close the information window"))
        self.close_pbutton.setText(_translate("info_window", "Close"))
