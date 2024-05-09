"""Tree window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow, QTreeWidgetItem

# %% Internal package import

from pyanno4rt.gui.compilations.tree_window import Ui_tree_window
from pyanno4rt.gui.windows import TextWindow

# %% Class definition


class TreeWindow(QMainWindow, Ui_tree_window):
    """
    Tree window for the application.

    This class creates a tree window for the graphical user interface,
    including a tree-based table view for dictionaries.
    """

    def __init__(
            self,
            title,
            parent=None):

        # Get the application from the argument
        self.parent = parent

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # 
        self.setWindowTitle(title)

        # 
        self.text_window = TextWindow(self)

        # 
        self.tree_widget.itemDoubleClicked.connect(self.show_item_text)

        # 
        self.expand_tree_pbutton.clicked.connect(self.expand_all)

        # 
        self.collapse_tree_pbutton.clicked.connect(self.collapse_all)

        # 
        self.close_tree_pbutton.clicked.connect(self.close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def create_tree_from_dict(self, data=None, parent=None):
        """."""

        # 
        for key, value in data.items():

            # 
            item = QTreeWidgetItem(parent)

            # 
            item.setText(0, key)

            # 
            if isinstance(value, dict):

                # 
                self.create_tree_from_dict(data=value, parent=item)

            else:
                item.setText(1, type(value).__name__)
                item.setText(2, str(value))

    def show_item_text(self, tree, item):
        """."""

        # 
        self.text_window.text_tedit.clear()

        # 
        self.text_window.position()

        # 
        self.text_window.text_tedit.setText(tree.text(item))

        # 
        self.text_window.show()

    def expand_all(self):
        """."""

        # 
        self.tree_widget.expandAll()

    def collapse_all(self):
        """."""

        # 
        self.tree_widget.collapseAll()

    def close(self):
        """."""

        # 
        self.hide()
