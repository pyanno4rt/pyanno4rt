# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'equivalent_uniform_dose_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_equivalent_uniform_dose_window(object):
    def setupUi(self, equivalent_uniform_dose_window):
        equivalent_uniform_dose_window.setObjectName("equivalent_uniform_dose_window")
        equivalent_uniform_dose_window.setWindowModality(QtCore.Qt.ApplicationModal)
        equivalent_uniform_dose_window.resize(720, 431)
        equivalent_uniform_dose_window.setMinimumSize(QtCore.QSize(720, 431))
        equivalent_uniform_dose_window.setMaximumSize(QtCore.QSize(720, 431))
        equivalent_uniform_dose_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.body_widget = QtWidgets.QWidget(equivalent_uniform_dose_window)
        self.body_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.body_widget.setObjectName("body_widget")
        self.scroll_area = QtWidgets.QScrollArea(self.body_widget)
        self.scroll_area.setGeometry(QtCore.QRect(10, 0, 711, 431))
        self.scroll_area.setStyleSheet("border-color: transparent;")
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_contents = QtWidgets.QWidget()
        self.scroll_contents.setGeometry(QtCore.QRect(0, 0, 690, 681))
        self.scroll_contents.setObjectName("scroll_contents")
        self.embedding_cbox = QtWidgets.QComboBox(self.scroll_contents)
        self.embedding_cbox.setGeometry(QtCore.QRect(574, 100, 111, 31))
        self.embedding_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.embedding_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.embedding_cbox.setObjectName("embedding_cbox")
        self.embedding_cbox.addItem("")
        self.embedding_cbox.addItem("")
        self.segment_label = QtWidgets.QLabel(self.scroll_contents)
        self.segment_label.setGeometry(QtCore.QRect(0, 60, 431, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.segment_label.setFont(font)
        self.segment_label.setStyleSheet("border: 0px solid;")
        self.segment_label.setObjectName("segment_label")
        self.embedding_label = QtWidgets.QLabel(self.scroll_contents)
        self.embedding_label.setGeometry(QtCore.QRect(575, 60, 109, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.embedding_label.setFont(font)
        self.embedding_label.setStyleSheet("border: 0px solid;")
        self.embedding_label.setObjectName("embedding_label")
        self.base_label = QtWidgets.QLabel(self.scroll_contents)
        self.base_label.setGeometry(QtCore.QRect(0, 10, 684, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.base_label.setFont(font)
        self.base_label.setStyleSheet("border: 0px solid;")
        self.base_label.setObjectName("base_label")
        self.type_cbox = QtWidgets.QComboBox(self.scroll_contents)
        self.type_cbox.setGeometry(QtCore.QRect(445, 100, 111, 31))
        self.type_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.type_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.type_cbox.setObjectName("type_cbox")
        self.type_cbox.addItem("")
        self.type_cbox.addItem("")
        self.base_line = QtWidgets.QFrame(self.scroll_contents)
        self.base_line.setGeometry(QtCore.QRect(0, 50, 684, 2))
        self.base_line.setStyleSheet("border-color: rgb(46, 52, 54);")
        self.base_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.base_line.setObjectName("base_line")
        self.type_label = QtWidgets.QLabel(self.scroll_contents)
        self.type_label.setGeometry(QtCore.QRect(445, 60, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.type_label.setFont(font)
        self.type_label.setStyleSheet("border: 0px solid;")
        self.type_label.setObjectName("type_label")
        self.close_component_pbutton = QtWidgets.QPushButton(self.scroll_contents)
        self.close_component_pbutton.setGeometry(QtCore.QRect(614, 640, 71, 31))
        self.close_component_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_component_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_component_pbutton.setObjectName("close_component_pbutton")
        self.target_eud_label = QtWidgets.QLabel(self.scroll_contents)
        self.target_eud_label.setGeometry(QtCore.QRect(0, 210, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.target_eud_label.setFont(font)
        self.target_eud_label.setStyleSheet("border: 0px solid;")
        self.target_eud_label.setObjectName("target_eud_label")
        self.identifier_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.identifier_ledit.setGeometry(QtCore.QRect(0, 550, 381, 31))
        self.identifier_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.identifier_ledit.setObjectName("identifier_ledit")
        self.rank_sbox = QtWidgets.QSpinBox(self.scroll_contents)
        self.rank_sbox.setGeometry(QtCore.QRect(190, 400, 111, 31))
        self.rank_sbox.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.rank_sbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.rank_sbox.setAlignment(QtCore.Qt.AlignCenter)
        self.rank_sbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.rank_sbox.setMinimum(1)
        self.rank_sbox.setMaximum(1000)
        self.rank_sbox.setProperty("value", 1)
        self.rank_sbox.setObjectName("rank_sbox")
        self.identifier_label = QtWidgets.QLabel(self.scroll_contents)
        self.identifier_label.setGeometry(QtCore.QRect(0, 510, 381, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.identifier_label.setFont(font)
        self.identifier_label.setStyleSheet("border: 0px solid;")
        self.identifier_label.setObjectName("identifier_label")
        self.link_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.link_ledit.setGeometry(QtCore.QRect(290, 250, 394, 31))
        self.link_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.link_ledit.setObjectName("link_ledit")
        self.link_label = QtWidgets.QLabel(self.scroll_contents)
        self.link_label.setGeometry(QtCore.QRect(290, 210, 394, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.link_label.setFont(font)
        self.link_label.setStyleSheet("border: none;")
        self.link_label.setObjectName("link_label")
        self.function_label = QtWidgets.QLabel(self.scroll_contents)
        self.function_label.setGeometry(QtCore.QRect(0, 160, 684, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.function_label.setFont(font)
        self.function_label.setStyleSheet("border: 0px solid;")
        self.function_label.setObjectName("function_label")
        self.bounds_label = QtWidgets.QLabel(self.scroll_contents)
        self.bounds_label.setGeometry(QtCore.QRect(323, 360, 364, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bounds_label.setFont(font)
        self.bounds_label.setStyleSheet("border: 0px solid;")
        self.bounds_label.setObjectName("bounds_label")
        self.handling_viz_label = QtWidgets.QLabel(self.scroll_contents)
        self.handling_viz_label.setGeometry(QtCore.QRect(0, 460, 684, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.handling_viz_label.setFont(font)
        self.handling_viz_label.setStyleSheet("border: 0px solid;")
        self.handling_viz_label.setObjectName("handling_viz_label")
        self.target_eud_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.target_eud_ledit.setGeometry(QtCore.QRect(0, 250, 131, 31))
        self.target_eud_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.target_eud_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.target_eud_ledit.setObjectName("target_eud_ledit")
        self.weight_label = QtWidgets.QLabel(self.scroll_contents)
        self.weight_label.setGeometry(QtCore.QRect(0, 360, 171, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.weight_label.setFont(font)
        self.weight_label.setStyleSheet("border: 0px solid;")
        self.weight_label.setObjectName("weight_label")
        self.save_component_pbutton = QtWidgets.QPushButton(self.scroll_contents)
        self.save_component_pbutton.setGeometry(QtCore.QRect(534, 640, 71, 31))
        self.save_component_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_component_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.save_component_pbutton.setObjectName("save_component_pbutton")
        self.function_line = QtWidgets.QFrame(self.scroll_contents)
        self.function_line.setGeometry(QtCore.QRect(0, 200, 684, 2))
        self.function_line.setStyleSheet("border-color: rgb(46, 52, 54);")
        self.function_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.function_line.setObjectName("function_line")
        self.conjunction = QtWidgets.QLabel(self.scroll_contents)
        self.conjunction.setGeometry(QtCore.QRect(494, 405, 16, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.conjunction.setFont(font)
        self.conjunction.setStyleSheet("border: none;")
        self.conjunction.setAlignment(QtCore.Qt.AlignCenter)
        self.conjunction.setObjectName("conjunction")
        self.optimization_label = QtWidgets.QLabel(self.scroll_contents)
        self.optimization_label.setGeometry(QtCore.QRect(0, 310, 684, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.optimization_label.setFont(font)
        self.optimization_label.setStyleSheet("border: 0px solid;")
        self.optimization_label.setObjectName("optimization_label")
        self.rank_label = QtWidgets.QLabel(self.scroll_contents)
        self.rank_label.setGeometry(QtCore.QRect(190, 360, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.rank_label.setFont(font)
        self.rank_label.setStyleSheet("border: 0px solid;")
        self.rank_label.setObjectName("rank_label")
        self.upper_bound_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.upper_bound_ledit.setGeometry(QtCore.QRect(515, 400, 170, 31))
        self.upper_bound_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.upper_bound_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.upper_bound_ledit.setObjectName("upper_bound_ledit")
        self.weight_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.weight_ledit.setGeometry(QtCore.QRect(0, 400, 171, 31))
        self.weight_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.weight_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.weight_ledit.setObjectName("weight_ledit")
        self.disp_component_check = QtWidgets.QCheckBox(self.scroll_contents)
        self.disp_component_check.setGeometry(QtCore.QRect(0, 600, 231, 23))
        self.disp_component_check.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.disp_component_check.setStyleSheet("border: 0px;\n"
"color: rgb(0, 0, 0);")
        self.disp_component_check.setChecked(True)
        self.disp_component_check.setObjectName("disp_component_check")
        self.handling_viz_line = QtWidgets.QFrame(self.scroll_contents)
        self.handling_viz_line.setGeometry(QtCore.QRect(0, 500, 684, 2))
        self.handling_viz_line.setStyleSheet("border-color: rgb(46, 52, 54);")
        self.handling_viz_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.handling_viz_line.setObjectName("handling_viz_line")
        self.optimization_line = QtWidgets.QFrame(self.scroll_contents)
        self.optimization_line.setGeometry(QtCore.QRect(0, 350, 684, 2))
        self.optimization_line.setStyleSheet("border-color: rgb(46, 52, 54);")
        self.optimization_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.optimization_line.setObjectName("optimization_line")
        self.lower_bound_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.lower_bound_ledit.setGeometry(QtCore.QRect(320, 400, 170, 31))
        self.lower_bound_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.lower_bound_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.lower_bound_ledit.setObjectName("lower_bound_ledit")
        self.vol_eff_ledit = QtWidgets.QLineEdit(self.scroll_contents)
        self.vol_eff_ledit.setGeometry(QtCore.QRect(150, 250, 123, 31))
        self.vol_eff_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.vol_eff_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.vol_eff_ledit.setObjectName("vol_eff_ledit")
        self.vol_eff_label = QtWidgets.QLabel(self.scroll_contents)
        self.vol_eff_label.setGeometry(QtCore.QRect(150, 210, 123, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.vol_eff_label.setFont(font)
        self.vol_eff_label.setStyleSheet("border: 0px solid;")
        self.vol_eff_label.setObjectName("vol_eff_label")
        self.segment_cbox = QtWidgets.QComboBox(self.scroll_contents)
        self.segment_cbox.setGeometry(QtCore.QRect(0, 100, 426, 31))
        self.segment_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.segment_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.segment_cbox.setObjectName("segment_cbox")
        self.scroll_area.setWidget(self.scroll_contents)
        equivalent_uniform_dose_window.setCentralWidget(self.body_widget)

        self.retranslateUi(equivalent_uniform_dose_window)
        self.embedding_cbox.setCurrentIndex(0)
        self.type_cbox.setCurrentIndex(0)
        self.segment_cbox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(equivalent_uniform_dose_window)

    def retranslateUi(self, equivalent_uniform_dose_window):
        _translate = QtCore.QCoreApplication.translate
        equivalent_uniform_dose_window.setWindowTitle(_translate("equivalent_uniform_dose_window", "Equivalent Uniform Dose Editor"))
        self.embedding_cbox.setItemText(0, _translate("equivalent_uniform_dose_window", "active"))
        self.embedding_cbox.setItemText(1, _translate("equivalent_uniform_dose_window", "passive"))
        self.segment_label.setText(_translate("equivalent_uniform_dose_window", "Segment"))
        self.embedding_label.setText(_translate("equivalent_uniform_dose_window", "Embedding"))
        self.base_label.setText(_translate("equivalent_uniform_dose_window", "Base"))
        self.type_cbox.setItemText(0, _translate("equivalent_uniform_dose_window", "objective"))
        self.type_cbox.setItemText(1, _translate("equivalent_uniform_dose_window", "constraint"))
        self.type_label.setText(_translate("equivalent_uniform_dose_window", "Type"))
        self.close_component_pbutton.setText(_translate("equivalent_uniform_dose_window", "Close"))
        self.target_eud_label.setText(_translate("equivalent_uniform_dose_window", "Target EUD [Gy]"))
        self.identifier_ledit.setPlaceholderText(_translate("equivalent_uniform_dose_window", "None"))
        self.identifier_label.setText(_translate("equivalent_uniform_dose_window", "Identifier"))
        self.link_ledit.setPlaceholderText(_translate("equivalent_uniform_dose_window", "None"))
        self.link_label.setText(_translate("equivalent_uniform_dose_window", "Link"))
        self.function_label.setText(_translate("equivalent_uniform_dose_window", "Function"))
        self.bounds_label.setText(_translate("equivalent_uniform_dose_window", "Bounds"))
        self.handling_viz_label.setText(_translate("equivalent_uniform_dose_window", "Handling & Visualization"))
        self.weight_label.setText(_translate("equivalent_uniform_dose_window", "Weight"))
        self.save_component_pbutton.setText(_translate("equivalent_uniform_dose_window", "Save"))
        self.conjunction.setText(_translate("equivalent_uniform_dose_window", "-"))
        self.optimization_label.setText(_translate("equivalent_uniform_dose_window", "Optimization"))
        self.rank_label.setText(_translate("equivalent_uniform_dose_window", "Rank"))
        self.upper_bound_ledit.setPlaceholderText(_translate("equivalent_uniform_dose_window", "None"))
        self.weight_ledit.setPlaceholderText(_translate("equivalent_uniform_dose_window", "1.0"))
        self.disp_component_check.setText(_translate("equivalent_uniform_dose_window", "Display component"))
        self.lower_bound_ledit.setPlaceholderText(_translate("equivalent_uniform_dose_window", "0.0"))
        self.vol_eff_label.setText(_translate("equivalent_uniform_dose_window", "Volume effect"))
