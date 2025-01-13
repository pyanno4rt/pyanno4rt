"""Tree window."""

# Author: Tim Ortkamp

# %% External package import

from PyQt5.QtWidgets import QMainWindow, QTreeWidgetItem

# %% Internal package import

from pyanno4rt.gui.compilations.tree_window import Ui_tree_window
from pyanno4rt.gui.styles._custom_styles import pbutton_composer
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

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Get the application from the argument
        self.parent = parent

        # 
        self.setWindowTitle(title)

        # 
        self.text_window = TextWindow(self)

        # Set the stylesheets
        self.set_styles({
            'expand_tree_pbutton': pbutton_composer,
            'collapse_tree_pbutton': pbutton_composer,
            'close_tree_pbutton': pbutton_composer})

        # 
        self.connect_signals()

    def set_styles(
            self,
            key_value_pairs):
        """
        Set the element stylesheets from key-value pairs.

        Parameters
        ----------
        key_value_pairs : dict
            Dictionary with the field names (keys) and style sheets (values).
        """

        # Loop over the dictionary items
        for key, value in key_value_pairs.items():

            # Get the attribute and set the stylesheet
            getattr(self, key).setStyleSheet(value)

    def connect_signals(self):
        """Connect the fields with the event signals."""

        # Loop over the field names with 'clicked' events
        for key, value in {
            'expand_tree_pbutton': self.expand,
            'collapse_tree_pbutton': self.collapse,
            'close_tree_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # 
        self.tree_widget.itemDoubleClicked.connect(self.show_item_text)

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

    def expand(self):
        """."""

        # 
        self.tree_widget.expandAll()

    def collapse(self):
        """."""

        # 
        self.tree_widget.collapseAll()

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the tree window."""

        # Hide the window
        self.hide()
