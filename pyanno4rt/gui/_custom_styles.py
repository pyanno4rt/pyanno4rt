"""Custom style definitions."""

# Author: Tim Ortkamp

# %% Style definitions

# Set the stylesheet for the standard combo box fields
cbox = ('''
        QComboBox {
            color: rgb(0, 0, 0);
            background-color: rgb(238, 238, 236);
            border: 1px solid;
            border-color: rgb(186, 189, 182);
            }
        QComboBox QAbstractItemView {
            color: rgb(0, 0, 0);
            background-color: rgb(238, 238, 236);
            }
        QComboBox:disabled {
            color: rgb(153, 153, 153);
            }
        ''')

# Set the stylesheet for the line edit fields
ledit = ('''
         QLineEdit {
             color: rgb(0, 0, 0);
             background-color: rgb(238, 238, 236);
             border: 1px solid;
             border-color: rgb(186, 189, 182);
             }
         QLineEdit:disabled {
             color: rgb(153, 153, 153);
             }
         ''')

# Set the stylesheet for the menu push buttons
pbutton_menu = ('''
                QPushButton {
                    background-color: rgb(211, 215, 207);
                    color: rgb(0, 0, 0);
                    border: 1px solid;
                    border-color: rgb(186, 189, 182);
                    }
                QPushButton:disabled {
                    color: rgb(153, 153, 153);
                    }
                QPushButton:hover {
                    background-color: rgb(246, 246, 244);
                    }
                ''')

# Set the stylesheet for the composer update/reset buttons
pbutton_composer = ('''
                    QPushButton {
                        color: rgb(0, 0, 0);
                        background-color: rgb(238, 238, 236);
                        border: 1px solid;
                        border-color: rgb(186, 189, 182);
                        }
                    QPushButton:disabled {
                        color: rgb(153, 153, 153);
                        }
                    QPushButton:hover {
                        background-color: rgb(246, 246, 244);
                        }
                    ''')

# Set the stylesheet for the status bar push buttons
pbutton_statusbar = ('''
                     QPushButton {
                         border: 0px;
                         }
                     QPushButton:hover {
                         background-color: rgb(30, 30, 30);
                         }
                     ''')

# Set the stylesheet for the workflow push buttons
pbutton_workflow = ('''
                    QPushButton {
                        color: rgb(0, 0, 0);
                        background-color: rgb(211, 215, 207);
                        border: 1px solid;
                        border-color: rgb(186, 189, 182);
                        }
                    QPushButton:disabled {
                        color: rgb(153, 153, 153);
                        }
                    QPushButton:hover {
                        background-color: rgb(246, 246, 244);
                        }
                    ''')

# Set the stylesheet for the standard spin box fields
sbox = ('''
        QSpinBox {
            color: rgb(0, 0, 0);
            background-color: rgb(238, 238, 236);
            border: 1px solid;
            border-color: rgb(186, 189, 182);
            }
        QSpinBox QAbstractItemView {
            color: rgb(0, 0, 0);
            background-color: rgb(238, 238, 236);
            }
        QSpinBox:disabled {
            color: rgb(153, 153, 153);
            }
        ''')

# Set the stylesheet for the selector
selector = ('''
            QComboBox {
                color: rgb(211, 50, 50);
                background-color: rgb(0, 0, 0);
                border: 1px solid;
                border-color: rgb(186, 189, 182);
                }
            QComboBox::item {
                color: rgb(211, 50, 50);
                background-color: rgb(0, 0, 0);
                border: 1px solid;
                border-color: rgb(186, 189, 182);
                min-height: 36px;
                max-height: 36px;
                }
            QComboBox::item:selected {
                background-color: rgb(32, 23, 23);
                border: 1px solid;
                border-color: rgb(186, 189, 182);
                }
            ''')

# Set the stylesheet for the tab bars
tab = ('''
       QTabBar::tab:selected {
           background-color: rgb(30, 30, 30);
           }
       ''')

# Set the stylesheet for the composer tool buttons
tbutton_composer = ('''
                    QToolButton {
                        color: rgb(0, 0, 0);
                        background-color: rgb(238, 238, 236);
                        border: 1px solid;
                        border-color: rgb(186, 189, 182);
                        }
                    QToolButton:disabled {
                        color: rgb(153, 153, 153);
                        }
                    QToolButton:hover {
                        background-color: rgb(246, 246, 244);
                        }
                    QToolButton::menu-indicator {
                        image: none;
                        }
                    ''')

# Set the stylesheet for the data columns window tool button
tbutton_data_window = ('''
                     QToolButton {
                         background-color: rgb(0, 0, 0);
                         border: 1px solid;
                         border-color: rgb(46, 52, 54);
                         color: rgb(238, 238, 236);
                         }
                     QToolButton:hover {
                         background-color: rgb(30, 30, 30);
                         }
                     ''')

# Set the stylesheet for the workflow tool buttons
tbutton_workflow = ('''
                    QToolButton {
                        color: rgb(0, 0, 0);
                        background-color: rgb(211, 215, 207);
                        border: 1px solid;
                        border-color: rgb(186, 189, 182);
                        }
                    QToolButton:disabled {
                        color: rgb(153, 153, 153);
                        }
                    QToolButton:hover {
                        background-color: rgb(246, 246, 244);
                        }
                    ''')

# Set the stylesheet for the tooltip
tooltip = ('''
           QToolTip {
               background-color: #1e1e1e;
               color: #dddddd;
               border-color: #1e1e1e;
               }
           ''')
