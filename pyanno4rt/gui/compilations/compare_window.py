# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_compare_window(object):
    def setupUi(self, compare_window):
        compare_window.setObjectName("compare_window")
        compare_window.setWindowModality(QtCore.Qt.WindowModal)
        compare_window.resize(790, 460)
        compare_window.setMinimumSize(QtCore.QSize(790, 460))
        compare_window.setMaximumSize(QtCore.QSize(790, 460))
        compare_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.compare_widget = QtWidgets.QWidget(compare_window)
        self.compare_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.compare_widget.setObjectName("compare_widget")
        self.close_info_pbutton = QtWidgets.QPushButton(self.compare_widget)
        self.close_info_pbutton.setGeometry(QtCore.QRect(710, 410, 71, 30))
        self.close_info_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_info_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_info_pbutton.setObjectName("close_info_pbutton")
        self.baseline_widget = QtWidgets.QWidget(self.compare_widget)
        self.baseline_widget.setGeometry(QtCore.QRect(0, 30, 390, 346))
        self.baseline_widget.setStyleSheet("border: 1px solid;\n"
"border-color: rgb(211, 215, 207);")
        self.baseline_widget.setObjectName("baseline_widget")
        self.baseline_meta_layout = QtWidgets.QVBoxLayout(self.baseline_widget)
        self.baseline_meta_layout.setContentsMargins(10, 10, 10, 0)
        self.baseline_meta_layout.setSpacing(0)
        self.baseline_meta_layout.setObjectName("baseline_meta_layout")
        self.baseline_layout = QtWidgets.QVBoxLayout()
        self.baseline_layout.setSpacing(3)
        self.baseline_layout.setObjectName("baseline_layout")
        self.baseline_meta_layout.addLayout(self.baseline_layout)
        self.slice_selection_sbar = QtWidgets.QScrollBar(self.compare_widget)
        self.slice_selection_sbar.setGeometry(QtCore.QRect(130, 410, 571, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_selection_sbar.sizePolicy().hasHeightForWidth())
        self.slice_selection_sbar.setSizePolicy(sizePolicy)
        self.slice_selection_sbar.setMinimumSize(QtCore.QSize(239, 30))
        self.slice_selection_sbar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.slice_selection_sbar.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.slice_selection_sbar.setProperty("value", 49)
        self.slice_selection_sbar.setOrientation(QtCore.Qt.Horizontal)
        self.slice_selection_sbar.setObjectName("slice_selection_sbar")
        self.reference_widget = QtWidgets.QWidget(self.compare_widget)
        self.reference_widget.setGeometry(QtCore.QRect(400, 30, 390, 346))
        self.reference_widget.setStyleSheet("border: 1px solid;\n"
"border-color: rgb(211, 215, 207);")
        self.reference_widget.setObjectName("reference_widget")
        self.reference_meta_layout = QtWidgets.QVBoxLayout(self.reference_widget)
        self.reference_meta_layout.setContentsMargins(10, 10, 10, 0)
        self.reference_meta_layout.setSpacing(0)
        self.reference_meta_layout.setObjectName("reference_meta_layout")
        self.reference_layout = QtWidgets.QVBoxLayout()
        self.reference_layout.setSpacing(3)
        self.reference_layout.setObjectName("reference_layout")
        self.reference_meta_layout.addLayout(self.reference_layout)
        self.baseline_label = QtWidgets.QLabel(self.compare_widget)
        self.baseline_label.setGeometry(QtCore.QRect(10, 10, 371, 25))
        self.baseline_label.setMinimumSize(QtCore.QSize(0, 25))
        self.baseline_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.baseline_label.setFont(font)
        self.baseline_label.setStyleSheet("border: 0px;")
        self.baseline_label.setAlignment(QtCore.Qt.AlignCenter)
        self.baseline_label.setObjectName("baseline_label")
        self.reference_label = QtWidgets.QLabel(self.compare_widget)
        self.reference_label.setGeometry(QtCore.QRect(410, 10, 371, 25))
        self.reference_label.setMinimumSize(QtCore.QSize(0, 25))
        self.reference_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.reference_label.setFont(font)
        self.reference_label.setStyleSheet("border: 0px;")
        self.reference_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reference_label.setObjectName("reference_label")
        self.slice_selection_label = QtWidgets.QLabel(self.compare_widget)
        self.slice_selection_label.setGeometry(QtCore.QRect(130, 380, 118, 25))
        self.slice_selection_label.setMinimumSize(QtCore.QSize(0, 25))
        self.slice_selection_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.slice_selection_label.setFont(font)
        self.slice_selection_label.setStyleSheet("border: 0px;")
        self.slice_selection_label.setObjectName("slice_selection_label")
        self.opacity_label = QtWidgets.QLabel(self.compare_widget)
        self.opacity_label.setGeometry(QtCore.QRect(10, 380, 120, 25))
        self.opacity_label.setMinimumSize(QtCore.QSize(0, 25))
        self.opacity_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.opacity_label.setFont(font)
        self.opacity_label.setStyleSheet("border: 0px;")
        self.opacity_label.setObjectName("opacity_label")
        self.opacity_sbox = QtWidgets.QSpinBox(self.compare_widget)
        self.opacity_sbox.setGeometry(QtCore.QRect(10, 410, 120, 30))
        self.opacity_sbox.setMinimumSize(QtCore.QSize(120, 30))
        self.opacity_sbox.setMaximumSize(QtCore.QSize(120, 30))
        self.opacity_sbox.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.opacity_sbox.setStyleSheet("background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);\n"
"margin-right: 6px;")
        self.opacity_sbox.setAlignment(QtCore.Qt.AlignCenter)
        self.opacity_sbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.opacity_sbox.setMaximum(100)
        self.opacity_sbox.setProperty("value", 70)
        self.opacity_sbox.setObjectName("opacity_sbox")
        self.slice_selection_pos = QtWidgets.QLabel(self.compare_widget)
        self.slice_selection_pos.setGeometry(QtCore.QRect(580, 380, 120, 25))
        self.slice_selection_pos.setMinimumSize(QtCore.QSize(0, 25))
        self.slice_selection_pos.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.slice_selection_pos.setFont(font)
        self.slice_selection_pos.setStyleSheet("border: 0px;")
        self.slice_selection_pos.setText("")
        self.slice_selection_pos.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.slice_selection_pos.setObjectName("slice_selection_pos")
        compare_window.setCentralWidget(self.compare_widget)

        self.retranslateUi(compare_window)
        QtCore.QMetaObject.connectSlotsByName(compare_window)

    def retranslateUi(self, compare_window):
        _translate = QtCore.QCoreApplication.translate
        compare_window.setWindowTitle(_translate("compare_window", "Plan comparison"))
        self.close_info_pbutton.setText(_translate("compare_window", "Close"))
        self.slice_selection_label.setText(_translate("compare_window", "Slice selection"))
        self.opacity_label.setText(_translate("compare_window", "Dose opacity"))
