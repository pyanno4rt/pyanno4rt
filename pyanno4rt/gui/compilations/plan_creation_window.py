# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plan_creation_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_plan_create_window(object):
    def setupUi(self, plan_create_window):
        plan_create_window.setObjectName("plan_create_window")
        plan_create_window.setWindowModality(QtCore.Qt.ApplicationModal)
        plan_create_window.resize(330, 240)
        plan_create_window.setMinimumSize(QtCore.QSize(330, 240))
        plan_create_window.setMaximumSize(QtCore.QSize(330, 240))
        plan_create_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.body_widget = QtWidgets.QWidget(plan_create_window)
        self.body_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.body_widget.setObjectName("body_widget")
        self.close_plan_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.close_plan_pbutton.setGeometry(QtCore.QRect(250, 200, 71, 31))
        self.close_plan_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_plan_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_plan_pbutton.setObjectName("close_plan_pbutton")
        self.create_plan_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.create_plan_pbutton.setGeometry(QtCore.QRect(170, 200, 71, 31))
        self.create_plan_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.create_plan_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.create_plan_pbutton.setObjectName("create_plan_pbutton")
        self.new_plan_label = QtWidgets.QLabel(self.body_widget)
        self.new_plan_label.setGeometry(QtCore.QRect(10, 20, 161, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_plan_label.setFont(font)
        self.new_plan_label.setStyleSheet("border: 0px solid;")
        self.new_plan_label.setObjectName("new_plan_label")
        self.new_plan_ref_label = QtWidgets.QLabel(self.body_widget)
        self.new_plan_ref_label.setGeometry(QtCore.QRect(10, 110, 161, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_plan_ref_label.setFont(font)
        self.new_plan_ref_label.setStyleSheet("border: 0px solid;")
        self.new_plan_ref_label.setObjectName("new_plan_ref_label")
        self.new_plan_ref_cbox = QtWidgets.QComboBox(self.body_widget)
        self.new_plan_ref_cbox.setGeometry(QtCore.QRect(10, 150, 311, 31))
        self.new_plan_ref_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_plan_ref_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_plan_ref_cbox.setObjectName("new_plan_ref_cbox")
        self.new_plan_ref_cbox.addItem("")
        self.new_plan_ledit = QtWidgets.QLineEdit(self.body_widget)
        self.new_plan_ledit.setGeometry(QtCore.QRect(10, 60, 311, 31))
        self.new_plan_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_plan_ledit.setObjectName("new_plan_ledit")
        plan_create_window.setCentralWidget(self.body_widget)

        self.retranslateUi(plan_create_window)
        self.new_plan_ref_cbox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(plan_create_window)

    def retranslateUi(self, plan_create_window):
        _translate = QtCore.QCoreApplication.translate
        plan_create_window.setWindowTitle(_translate("plan_create_window", "Plan Creator"))
        self.close_plan_pbutton.setText(_translate("plan_create_window", "Close"))
        self.create_plan_pbutton.setText(_translate("plan_create_window", "Create"))
        self.new_plan_label.setText(_translate("plan_create_window", "Treatment plan label"))
        self.new_plan_ref_label.setText(_translate("plan_create_window", "Reference plan"))
        self.new_plan_ref_cbox.setItemText(0, _translate("plan_create_window", "None"))
