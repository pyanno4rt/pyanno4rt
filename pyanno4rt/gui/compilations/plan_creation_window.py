# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plan_creation_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_plan_creation_window(object):
    def setupUi(self, plan_creation_window):
        plan_creation_window.setObjectName("plan_creation_window")
        plan_creation_window.setWindowModality(QtCore.Qt.ApplicationModal)
        plan_creation_window.resize(400, 600)
        plan_creation_window.setMinimumSize(QtCore.QSize(400, 600))
        plan_creation_window.setMaximumSize(QtCore.QSize(400, 600))
        plan_creation_window.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);")
        self.body_widget = QtWidgets.QWidget(plan_creation_window)
        self.body_widget.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(211, 215, 207);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.body_widget.setObjectName("body_widget")
        self.close_plan_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.close_plan_pbutton.setGeometry(QtCore.QRect(320, 560, 71, 31))
        self.close_plan_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_plan_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.close_plan_pbutton.setObjectName("close_plan_pbutton")
        self.create_plan_pbutton = QtWidgets.QPushButton(self.body_widget)
        self.create_plan_pbutton.setGeometry(QtCore.QRect(240, 560, 71, 31))
        self.create_plan_pbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.create_plan_pbutton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.create_plan_pbutton.setObjectName("create_plan_pbutton")
        self.new_plan_label = QtWidgets.QLabel(self.body_widget)
        self.new_plan_label.setGeometry(QtCore.QRect(10, 20, 381, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_plan_label.setFont(font)
        self.new_plan_label.setStyleSheet("border: 0px solid;")
        self.new_plan_label.setObjectName("new_plan_label")
        self.new_plan_ref_label = QtWidgets.QLabel(self.body_widget)
        self.new_plan_ref_label.setGeometry(QtCore.QRect(10, 110, 381, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_plan_ref_label.setFont(font)
        self.new_plan_ref_label.setStyleSheet("border: 0px solid;")
        self.new_plan_ref_label.setObjectName("new_plan_ref_label")
        self.new_plan_ref_cbox = QtWidgets.QComboBox(self.body_widget)
        self.new_plan_ref_cbox.setGeometry(QtCore.QRect(10, 150, 381, 31))
        self.new_plan_ref_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_plan_ref_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_plan_ref_cbox.setObjectName("new_plan_ref_cbox")
        self.new_plan_ref_cbox.addItem("")
        self.new_plan_ledit = QtWidgets.QLineEdit(self.body_widget)
        self.new_plan_ledit.setGeometry(QtCore.QRect(10, 60, 381, 31))
        self.new_plan_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_plan_ledit.setObjectName("new_plan_ledit")
        self.new_img_path_tbutton = QtWidgets.QToolButton(self.body_widget)
        self.new_img_path_tbutton.setGeometry(QtCore.QRect(360, 330, 31, 31))
        self.new_img_path_tbutton.setMinimumSize(QtCore.QSize(31, 31))
        self.new_img_path_tbutton.setMaximumSize(QtCore.QSize(31, 31))
        self.new_img_path_tbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_img_path_tbutton.setStyleSheet("background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/black_icons/icons_black/file-plus.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.new_img_path_tbutton.setIcon(icon)
        self.new_img_path_tbutton.setIconSize(QtCore.QSize(18, 18))
        self.new_img_path_tbutton.setObjectName("new_img_path_tbutton")
        self.new_img_path_label = QtWidgets.QLabel(self.body_widget)
        self.new_img_path_label.setGeometry(QtCore.QRect(10, 290, 381, 31))
        self.new_img_path_label.setMinimumSize(QtCore.QSize(0, 31))
        self.new_img_path_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_img_path_label.setFont(font)
        self.new_img_path_label.setStyleSheet("border: 0px solid;")
        self.new_img_path_label.setObjectName("new_img_path_label")
        self.new_img_path_ledit = QtWidgets.QLineEdit(self.body_widget)
        self.new_img_path_ledit.setGeometry(QtCore.QRect(10, 330, 341, 31))
        self.new_img_path_ledit.setMinimumSize(QtCore.QSize(0, 31))
        self.new_img_path_ledit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.new_img_path_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_img_path_ledit.setObjectName("new_img_path_ledit")
        self.new_dose_path_tbutton = QtWidgets.QToolButton(self.body_widget)
        self.new_dose_path_tbutton.setGeometry(QtCore.QRect(360, 420, 31, 31))
        self.new_dose_path_tbutton.setMinimumSize(QtCore.QSize(31, 31))
        self.new_dose_path_tbutton.setMaximumSize(QtCore.QSize(31, 31))
        self.new_dose_path_tbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_dose_path_tbutton.setStyleSheet("background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_dose_path_tbutton.setIcon(icon)
        self.new_dose_path_tbutton.setIconSize(QtCore.QSize(18, 18))
        self.new_dose_path_tbutton.setObjectName("new_dose_path_tbutton")
        self.new_dose_path_ledit = QtWidgets.QLineEdit(self.body_widget)
        self.new_dose_path_ledit.setGeometry(QtCore.QRect(10, 420, 341, 31))
        self.new_dose_path_ledit.setMinimumSize(QtCore.QSize(0, 31))
        self.new_dose_path_ledit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.new_dose_path_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_dose_path_ledit.setObjectName("new_dose_path_ledit")
        self.new_dose_path_label = QtWidgets.QLabel(self.body_widget)
        self.new_dose_path_label.setGeometry(QtCore.QRect(10, 380, 381, 31))
        self.new_dose_path_label.setMinimumSize(QtCore.QSize(0, 31))
        self.new_dose_path_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_dose_path_label.setFont(font)
        self.new_dose_path_label.setStyleSheet("border: 0px solid;")
        self.new_dose_path_label.setObjectName("new_dose_path_label")
        self.new_modality_label = QtWidgets.QLabel(self.body_widget)
        self.new_modality_label.setGeometry(QtCore.QRect(10, 200, 163, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.new_modality_label.sizePolicy().hasHeightForWidth())
        self.new_modality_label.setSizePolicy(sizePolicy)
        self.new_modality_label.setMinimumSize(QtCore.QSize(163, 31))
        self.new_modality_label.setMaximumSize(QtCore.QSize(163, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_modality_label.setFont(font)
        self.new_modality_label.setStyleSheet("border: 0px solid;")
        self.new_modality_label.setObjectName("new_modality_label")
        self.new_modality_cbox = QtWidgets.QComboBox(self.body_widget)
        self.new_modality_cbox.setGeometry(QtCore.QRect(10, 240, 163, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.new_modality_cbox.sizePolicy().hasHeightForWidth())
        self.new_modality_cbox.setSizePolicy(sizePolicy)
        self.new_modality_cbox.setMinimumSize(QtCore.QSize(163, 30))
        self.new_modality_cbox.setMaximumSize(QtCore.QSize(163, 30))
        self.new_modality_cbox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_modality_cbox.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_modality_cbox.setObjectName("new_modality_cbox")
        self.new_modality_cbox.addItem("")
        self.new_modality_cbox.addItem("")
        self.new_dose_res_ledit = QtWidgets.QLineEdit(self.body_widget)
        self.new_dose_res_ledit.setGeometry(QtCore.QRect(60, 510, 140, 30))
        self.new_dose_res_ledit.setMinimumSize(QtCore.QSize(140, 30))
        self.new_dose_res_ledit.setMaximumSize(QtCore.QSize(140, 30))
        self.new_dose_res_ledit.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(238, 238, 236);\n"
"border: 1px solid;\n"
"border-color: rgb(186, 189, 182);")
        self.new_dose_res_ledit.setAlignment(QtCore.Qt.AlignCenter)
        self.new_dose_res_ledit.setObjectName("new_dose_res_ledit")
        self.new_dose_res_label = QtWidgets.QLabel(self.body_widget)
        self.new_dose_res_label.setGeometry(QtCore.QRect(10, 470, 381, 31))
        self.new_dose_res_label.setMinimumSize(QtCore.QSize(0, 31))
        self.new_dose_res_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_dose_res_label.setFont(font)
        self.new_dose_res_label.setStyleSheet("border: 0px solid;")
        self.new_dose_res_label.setObjectName("new_dose_res_label")
        self.new_dose_res_sublabel = QtWidgets.QLabel(self.body_widget)
        self.new_dose_res_sublabel.setGeometry(QtCore.QRect(10, 510, 46, 30))
        self.new_dose_res_sublabel.setMinimumSize(QtCore.QSize(0, 30))
        self.new_dose_res_sublabel.setMaximumSize(QtCore.QSize(16777215, 30))
        self.new_dose_res_sublabel.setStyleSheet("border: 0px solid;")
        self.new_dose_res_sublabel.setObjectName("new_dose_res_sublabel")
        plan_creation_window.setCentralWidget(self.body_widget)

        self.retranslateUi(plan_creation_window)
        self.new_plan_ref_cbox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(plan_creation_window)

    def retranslateUi(self, plan_creation_window):
        _translate = QtCore.QCoreApplication.translate
        plan_creation_window.setWindowTitle(_translate("plan_creation_window", "Plan Creator"))
        self.close_plan_pbutton.setText(_translate("plan_creation_window", "Close"))
        self.create_plan_pbutton.setText(_translate("plan_creation_window", "Create"))
        self.new_plan_label.setText(_translate("plan_creation_window", "Treatment plan label"))
        self.new_plan_ref_label.setText(_translate("plan_creation_window", "Reference plan"))
        self.new_plan_ref_cbox.setItemText(0, _translate("plan_creation_window", "None"))
        self.new_img_path_tbutton.setToolTip(_translate("plan_creation_window", "Add the CT and segmentation data from a folder"))
        self.new_img_path_label.setText(_translate("plan_creation_window", "Imaging path"))
        self.new_dose_path_tbutton.setToolTip(_translate("plan_creation_window", "Add the dose-influence matrix from a folder"))
        self.new_dose_path_label.setText(_translate("plan_creation_window", "Dose matrix path"))
        self.new_modality_label.setText(_translate("plan_creation_window", "Modality"))
        self.new_modality_cbox.setItemText(0, _translate("plan_creation_window", "photon"))
        self.new_modality_cbox.setItemText(1, _translate("plan_creation_window", "proton"))
        self.new_dose_res_label.setToolTip(_translate("plan_creation_window", "Size of the dose grid in [mm] per dimension"))
        self.new_dose_res_label.setText(_translate("plan_creation_window", "Dose grid resolution [mm]"))
        self.new_dose_res_sublabel.setText(_translate("plan_creation_window", "[x, y, z]"))
