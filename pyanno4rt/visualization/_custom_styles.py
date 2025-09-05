"""Custom style definitions."""

# Author: Tim Ortkamp

# %% Style definitions

# Set the stylesheet for the standard combo box fields
cbox = ('''
        QComboBox {
            color: rgb(0, 0, 0);
            selection-color: rgb(0, 0, 0);
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

# Set the stylesheet for the push buttons
pbutton = ('''
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

# Set the stylesheets for the tab bars
tab_bright = ('''
              QTabBar::tab:selected {
                  background-color: rgb(238, 238, 236);
                  }
              ''')
tab_dark = ('''
            QTabBar::tab:selected {
                background-color: rgb(30, 30, 30);
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
