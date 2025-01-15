"""Data columns window."""

# Author: Tim Ortkamp

# %% External package import

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QLineEdit, QMainWindow,
    QSpinBox, QTableWidgetItem)

# %% Internal package import

from pyanno4rt.gui.compilations.data_columns_window import (
    Ui_data_columns_window)
from pyanno4rt.gui.styles._custom_styles import (
    cbox, pbutton_composer, tbutton_composer)
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.tools import apply, string_to_numeric

# %% Class definition


class DataColumnsWindow(QMainWindow, Ui_data_columns_window):
    """
    Data columns window for the components.

    This class creates the data columns window for the components in the \
    graphical user interface, including the features and labels.
    """

    def __init__(
            self,
            parent=None):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Get the application from the argument
        self.parent = parent

        # Set the stylesheets
        self.set_styles({
            'features_plus_tbutton': tbutton_composer,
            'features_minus_tbutton': tbutton_composer,
            'column_cbox': cbox,
            'viewpoint_cbox': cbox,
            'time_variable_cbox': cbox,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox and QSpinBox elements
        for box in (
                'features_table', 'column_cbox', 'viewpoint_cbox',
                'time_variable_cbox'):

            # Install the custom event filter
            getattr(self, box).wheelEvent = lambda event: None

        # 
        self.features_minus_tbutton.setEnabled(False)

        # Connect the event signals
        self.connect_signals()

    def mousePressEvent(
            self,
            event):
        """
        Set the mouse press event (overwrites the default event).

        Parameters
        ----------
        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.
        """

        # Reset the table index when clicking outside
        self.features_table.setCurrentIndex(QModelIndex())
        self.features_table.clearSelection()

        # 
        self.features_minus_tbutton.setEnabled(False)

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

            # Check if the key does not refer to a tab widget
            if key in ('composer_widget', 'tab_workflow', 'viewer_widget'):

                # Get the tab bar of the attribute and set the stylesheet
                getattr(self, key).tabBar().setStyleSheet(value)

            else:

                # Get the attribute and set the stylesheet
                getattr(self, key).setStyleSheet(value)

    def set_zero_line_cursor(
            self,
            field_names):
        """
        Set the line edit cursor positions to zero.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and set the cursor position to zero
            getattr(self, name).setCursorPosition(0)

    def set_enabled(
            self,
            field_names):
        """
        Enable multiple fields by their names.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and enable the field
            getattr(self, name).setEnabled(True)

    def set_disabled(
            self,
            field_names):
        """
        Disable multiple fields by their names.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and disable the field
            getattr(self, name).setEnabled(False)

    def connect_signals(self):
        """Connect the fields with the event signals."""

        # Loop over the field names with 'clicked' events
        for key, value in {
                'features_minus_tbutton': self.remove_feature,
                'save_pbutton': self.save,
                'close_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # 
        self.features_table.verticalHeader().sectionClicked.connect(
            self.select_row)

    def select_row(self, index):
        """."""

        # 
        self.features_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.features_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.features_table.selectRow(index)
        self.features_table.setSelectionMode(QAbstractItemView.NoSelection)

        # 
        self.features_minus_tbutton.setEnabled(True)

    def remove_feature(self):
        """Remove the selected feature."""

        # Remove the component item from the list widget
        self.features_table.removeRow(self.features_table.currentRow())

        # Disable some fields
        self.set_disabled(('features_minus_tbutton',))

    def load(self, data_columns):
        """."""

        # Set the initial number of rows and columns
        self.features_table.setRowCount(0)
        self.features_table.setColumnCount(5)

        # Add the horizontal header labels
        self.features_table.setHorizontalHeaderLabels(
            ['Scale', 'Segment', 'Function', 'Argument', 'Value'])

        # Load the feature values
        apply(self.add_feature_row, {
            key: value for key, value in data_columns.items()
            if value['type'] == 'feature'}.items())

        # 
        self.features_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        # Load the label values
        self.add_label({
            key: value for key, value in data_columns.items()
            if value['type'] == 'label'})

    def add_feature_row(
            self,
            feature_dict):
        """."""

        # 
        label, fields = feature_dict

        # 
        nrow = self.features_table.rowCount()

        # 
        self.features_table.insertRow(nrow)

        # 
        header = QTableWidgetItem()
        header.setText(label)
        self.features_table.setVerticalHeaderItem(nrow, header)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems(['metric', 'nominal', 'ordinal'])
        widget.setCurrentText(fields['scale'])
        self.features_table.setCellWidget(nrow, 0, widget)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems([
            self.parent.segment_cbox.itemText(i)
            for i in range(self.parent.segment_cbox.count())])
        widget.adjustSize()
        widget.setCurrentText(fields['segment'])
        self.features_table.setCellWidget(nrow, 1, widget)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems(list(feature_map.keys()))
        widget.adjustSize()
        widget.setCurrentText(fields['function'])
        self.features_table.setCellWidget(nrow, 2, widget)

        # 
        if fields['function'] in ('Dx', 'Vx'):

            # 
            widget = QSpinBox()
            widget.wheelEvent = lambda event: None
            widget.setRange(1, 99)
            widget.setValue(fields['argument'])

        # 
        elif fields['function'] == 'Dose Gradient':

            # 
            widget = QComboBox()
            widget.wheelEvent = lambda event: None
            widget.addItems(['x', 'y', 'z'])
            widget.setCurrentText(fields['argument'])

        # 
        elif fields['function'] == 'Dose Moment':

            # 
            widget = QLineEdit()
            widget.setText(fields['argument'])

        # 
        elif fields['function'] == 'Dose Subvolume':

            # 
            widget = QComboBox()
            widget.wheelEvent = lambda event: None
            widget.addItems([
                'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3'])
            widget.setCurrentText(fields['argument'])

        else:

            # 
            widget = QLineEdit()
            widget.setText('')

        # 
        self.features_table.setCellWidget(nrow, 3, widget)

        # 
        widget = QLineEdit()
        widget.setText(fields['value'])
        self.features_table.setCellWidget(nrow, 4, widget)

    def add_label(
            self,
            label_dict):
        """."""

        # 
        label_name = next(iter(label_dict))
        label_value = label_dict[label_name]

        # 
        self.column_cbox.clear()
        self.column_cbox.addItems(self.parent.column_names)
        self.column_cbox.setCurrentText(label_name)

        # 
        self.viewpoint_cbox.setCurrentText(label_value['viewpoint'])

        # 
        self.time_variable_cbox.clear()
        self.time_variable_cbox.addItems([''] + self.parent.column_names)
        self.time_variable_cbox.setCurrentText(
            '' if not label_value['time_variable']
            else label_value['time_variable'])

        # 
        self.lower_bound_ledit.setText(
            '' if not label_value['bounds'] or label_value['bounds'][0] == 1.0
            else str(label_value['bounds'][0]))
        self.upper_bound_ledit.setText(
            '' if not label_value['bounds'] or label_value['bounds'][1] == 1.0
            else str(label_value['bounds'][1]))

    def save(self):
        """."""

        for row in range(self.features_table.rowCount()):
            print(self.features_table.horizontalHeaderItem(row).text())
            for col in range(self.features_table.columnCount()):
                item = self.features_table.item(row, col)
                print(item)

        # 
        features = {}

        # 
        label = {
            self.column_cbox.currentText(): {
                'type': 'label',
                'viewpoint': self.viewpoint_cbox.currentText(),
                'time_variable': (
                    None if self.time_variable_cbox.currentText() == ''
                    else self.time_variable_cbox.currentText()),
                'bounds': [
                    1 if bound == '' else None if bound == 'None'
                    else string_to_numeric(bound) for bound in (
                        self.lower_bound_ledit.text(),
                        self.upper_bound_ledit.text())]}}

        # 
        self.parent.data_columns = features | label

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the log window."""

        # Hide the window
        self.hide()
