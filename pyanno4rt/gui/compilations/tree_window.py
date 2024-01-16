# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tree_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_tree_window(object):
    def setupUi(self, tree_window):
        tree_window.setObjectName("tree_window")
        tree_window.setWindowModality(QtCore.Qt.ApplicationModal)
        tree_window.resize(720, 640)
        tree_window.setMinimumSize(QtCore.QSize(720, 640))
        tree_window.setMaximumSize(QtCore.QSize(720, 640))
        tree_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.body_widget = QtWidgets.QWidget(tree_window)
        self.body_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.body_widget.setObjectName("body_widget")
        self.close_tree_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.close_tree_pbutton.setGeometry(QtCore.QRect(640, 600, 71, 31))
        self.close_tree_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_tree_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_tree_pbutton.setObjectName("close_tree_pbutton")
        self.tree_widget = QtWidgets.QTreeWidget(self.body_widget)
        self.tree_widget.setGeometry(QtCore.QRect(10, 10, 701, 581))
        self.tree_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.tree_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.tree_widget.setUniformRowHeights(True)
        self.tree_widget.setColumnCount(3)
        self.tree_widget.setObjectName("tree_widget")
        self.collapse_tree_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.collapse_tree_pbutton.setGeometry(QtCore.QRect(560, 600, 71, 31))
        self.collapse_tree_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.collapse_tree_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.collapse_tree_pbutton.setObjectName("collapse_tree_pbutton")
        self.expand_tree_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.expand_tree_pbutton.setGeometry(QtCore.QRect(480, 600, 71, 31))
        self.expand_tree_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.expand_tree_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.expand_tree_pbutton.setObjectName("expand_tree_pbutton")
        tree_window.setCentralWidget(self.body_widget)

        self.retranslateUi(tree_window)
        QtCore.QMetaObject.connectSlotsByName(tree_window)

    def retranslateUi(self, tree_window):
        _translate = QtCore.QCoreApplication.translate
        tree_window.setWindowTitle(_translate("tree_window", "Tree Viewer"))
        self.close_tree_pbutton.setText(_translate("tree_window", "Close"))
        self.tree_widget.headerItem().setText(0, _translate("tree_window", "Key"))
        self.tree_widget.headerItem().setText(1, _translate("tree_window", "Type"))
        self.tree_widget.headerItem().setText(2, _translate("tree_window", "Value"))
        self.collapse_tree_pbutton.setText(_translate("tree_window", "Collapse"))
        self.expand_tree_pbutton.setText(_translate("tree_window", "Expand"))
