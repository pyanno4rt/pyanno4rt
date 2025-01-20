# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data_columns_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_data_columns_window(object):
    def setupUi(self, data_columns_window):
        data_columns_window.setObjectName("data_columns_window")
        data_columns_window.setWindowModality(QtCore.Qt.ApplicationModal)
        data_columns_window.resize(920, 680)
        data_columns_window.setMinimumSize(QtCore.QSize(920, 680))
        data_columns_window.setMaximumSize(QtCore.QSize(920, 680))
        data_columns_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.data_columns_widget = QtWidgets.QWidget(data_columns_window)
        self.data_columns_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.data_columns_widget.setObjectName("data_columns_widget")
        self.close_pbutton = QtWidgets.QPushButton(self.data_columns_widget)
        self.close_pbutton.setGeometry(QtCore.QRect(840, 640, 71, 31))
        self.close_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_pbutton.setObjectName("close_pbutton")
        self.save_pbutton = QtWidgets.QPushButton(self.data_columns_widget)
        self.save_pbutton.setGeometry(QtCore.QRect(760, 640, 71, 31))
        self.save_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.save_pbutton.setObjectName("save_pbutton")
        self.feature_table = QtWidgets.QTableWidget(self.data_columns_widget)
        self.feature_table.setGeometry(QtCore.QRect(10, 50, 901, 411))
        self.feature_table.setMinimumSize(QtCore.QSize(901, 0))
        self.feature_table.setMaximumSize(QtCore.QSize(901, 16777215))
        self.feature_table.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.feature_table.setAlternatingRowColors(True)
        self.feature_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.feature_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.feature_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.feature_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.feature_table.setObjectName("feature_table")
        self.feature_table.setColumnCount(0)
        self.feature_table.setRowCount(0)
        self.feature_table.horizontalHeader().setMinimumSectionSize(0)
        self.feature_table.verticalHeader().setDefaultSectionSize(30)
        self.feature_table.verticalHeader().setMinimumSectionSize(30)
        self.feature_label = QtWidgets.QLabel(self.data_columns_widget)
        self.feature_label.setGeometry(QtCore.QRect(10, 10, 111, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.feature_label.setFont(font)
        self.feature_label.setStyleSheet("border: 0px solid;")
        self.feature_label.setObjectName("feature_label")
        self.label_label = QtWidgets.QLabel(self.data_columns_widget)
        self.label_label.setGeometry(QtCore.QRect(10, 520, 111, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_label.setFont(font)
        self.label_label.setStyleSheet("border: 0px solid;")
        self.label_label.setObjectName("label_label")
        self.feature_plus_tbutton = QtWidgets.QToolButton(self.data_columns_widget)
        self.feature_plus_tbutton.setGeometry(QtCore.QRect(10, 470, 31, 31))
        self.feature_plus_tbutton.setMinimumSize(QtCore.QSize(31, 31))
        self.feature_plus_tbutton.setMaximumSize(QtCore.QSize(31, 31))
        self.feature_plus_tbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.feature_plus_tbutton.setStyleSheet("background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/black_icons/icons_black/plus.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.feature_plus_tbutton.setIcon(icon)
        self.feature_plus_tbutton.setIconSize(QtCore.QSize(18, 18))
        self.feature_plus_tbutton.setObjectName("feature_plus_tbutton")
        self.feature_minus_tbutton = QtWidgets.QToolButton(self.data_columns_widget)
        self.feature_minus_tbutton.setGeometry(QtCore.QRect(50, 470, 31, 31))
        self.feature_minus_tbutton.setMinimumSize(QtCore.QSize(31, 31))
        self.feature_minus_tbutton.setMaximumSize(QtCore.QSize(31, 31))
        self.feature_minus_tbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.feature_minus_tbutton.setStyleSheet("background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/black_icons/icons_black/minus.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.feature_minus_tbutton.setIcon(icon1)
        self.feature_minus_tbutton.setIconSize(QtCore.QSize(18, 18))
        self.feature_minus_tbutton.setObjectName("feature_minus_tbutton")
        self.column_cbox = QtWidgets.QComboBox(self.data_columns_widget)
        self.column_cbox.setGeometry(QtCore.QRect(10, 590, 271, 31))
        self.column_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.column_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.column_cbox.setMaxVisibleItems(25)
        self.column_cbox.setObjectName("column_cbox")
        self.column_label = QtWidgets.QLabel(self.data_columns_widget)
        self.column_label.setGeometry(QtCore.QRect(10, 560, 271, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.column_label.setFont(font)
        self.column_label.setStyleSheet("border: 0px solid;")
        self.column_label.setObjectName("column_label")
        self.viewpoint_cbox = QtWidgets.QComboBox(self.data_columns_widget)
        self.viewpoint_cbox.setGeometry(QtCore.QRect(290, 590, 111, 31))
        self.viewpoint_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.viewpoint_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.viewpoint_cbox.setObjectName("viewpoint_cbox")
        self.viewpoint_cbox.addItem("")
        self.viewpoint_cbox.addItem("")
        self.viewpoint_cbox.addItem("")
        self.viewpoint_cbox.addItem("")
        self.viewpoint_cbox.addItem("")
        self.viewpoint_label = QtWidgets.QLabel(self.data_columns_widget)
        self.viewpoint_label.setGeometry(QtCore.QRect(290, 560, 111, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.viewpoint_label.setFont(font)
        self.viewpoint_label.setStyleSheet("border: 0px solid;")
        self.viewpoint_label.setObjectName("viewpoint_label")
        self.time_variable_cbox = QtWidgets.QComboBox(self.data_columns_widget)
        self.time_variable_cbox.setGeometry(QtCore.QRect(410, 590, 271, 31))
        self.time_variable_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.time_variable_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.time_variable_cbox.setMaxVisibleItems(25)
        self.time_variable_cbox.setObjectName("time_variable_cbox")
        self.time_variable_label = QtWidgets.QLabel(self.data_columns_widget)
        self.time_variable_label.setGeometry(QtCore.QRect(410, 560, 271, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_variable_label.setFont(font)
        self.time_variable_label.setStyleSheet("border: 0px solid;")
        self.time_variable_label.setObjectName("time_variable_label")
        self.bounds_label = QtWidgets.QLabel(self.data_columns_widget)
        self.bounds_label.setGeometry(QtCore.QRect(690, 560, 221, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bounds_label.setFont(font)
        self.bounds_label.setStyleSheet("border: 0px solid;")
        self.bounds_label.setObjectName("bounds_label")
        self.label_bounds_conjunction = QtWidgets.QLabel(self.data_columns_widget)
        self.label_bounds_conjunction.setGeometry(QtCore.QRect(793, 595, 16, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_bounds_conjunction.setFont(font)
        self.label_bounds_conjunction.setStyleSheet("border: none;")
        self.label_bounds_conjunction.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bounds_conjunction.setObjectName("label_bounds_conjunction")
        self.lower_bound_ledit = QtWidgets.QLineEdit(self.data_columns_widget)
        self.lower_bound_ledit.setGeometry(QtCore.QRect(690, 590, 101, 31))
        self.lower_bound_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.lower_bound_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.lower_bound_ledit.setObjectName("lower_bound_ledit")
        self.upper_bound_ledit = QtWidgets.QLineEdit(self.data_columns_widget)
        self.upper_bound_ledit.setGeometry(QtCore.QRect(810, 590, 101, 31))
        self.upper_bound_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.upper_bound_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.upper_bound_ledit.setObjectName("upper_bound_ledit")
        data_columns_window.setCentralWidget(self.data_columns_widget)

        self.retranslateUi(data_columns_window)
        self.column_cbox.setCurrentIndex(-1)
        self.viewpoint_cbox.setCurrentIndex(3)
        self.time_variable_cbox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(data_columns_window)

    def retranslateUi(self, data_columns_window):
        _translate = QtCore.QCoreApplication.translate
        data_columns_window.setWindowTitle(_translate("data_columns_window", "Data Columns Editor"))
        self.close_pbutton.setText(_translate("data_columns_window", "Close"))
        self.save_pbutton.setText(_translate("data_columns_window", "Save"))
        self.feature_label.setText(_translate("data_columns_window", "Features"))
        self.label_label.setText(_translate("data_columns_window", "Label"))
        self.feature_plus_tbutton.setToolTip(_translate("data_columns_window", "Add a component"))
        self.feature_minus_tbutton.setToolTip(_translate("data_columns_window", "Remove the selected component"))
        self.column_label.setText(_translate("data_columns_window", "Column"))
        self.viewpoint_cbox.setItemText(0, _translate("data_columns_window", "early"))
        self.viewpoint_cbox.setItemText(1, _translate("data_columns_window", "late"))
        self.viewpoint_cbox.setItemText(2, _translate("data_columns_window", "long-term"))
        self.viewpoint_cbox.setItemText(3, _translate("data_columns_window", "longitudinal"))
        self.viewpoint_cbox.setItemText(4, _translate("data_columns_window", "profile"))
        self.viewpoint_label.setText(_translate("data_columns_window", "Viewpoint"))
        self.time_variable_label.setText(_translate("data_columns_window", "Time variable"))
        self.bounds_label.setText(_translate("data_columns_window", "Bounds"))
        self.label_bounds_conjunction.setText(_translate("data_columns_window", "-"))
        self.lower_bound_ledit.setPlaceholderText(_translate("data_columns_window", "1"))
        self.upper_bound_ledit.setPlaceholderText(_translate("data_columns_window", "1"))
