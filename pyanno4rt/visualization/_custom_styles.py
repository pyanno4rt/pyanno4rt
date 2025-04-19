"""Custom style definitions."""

# Author: Tim Ortkamp

# %% Style definitions

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
