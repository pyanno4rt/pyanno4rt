"""Data columns window."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from json import load
from pandas import read_csv
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QLineEdit, QMainWindow,
    QMenu, QSpinBox, QTableWidgetItem)

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
    Data columns window for the machine learning model-based components.

    This class sets up the data columns window for the machine learning \
    model-based components in the graphical user interface, including the \
    features table and label fields.
    """

    def __init__(
            self,
            parent=None):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Get the parent window
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

        # Loop over the QComboBox elements
        for box in ('column_cbox', 'viewpoint_cbox', 'time_variable_cbox'):

            # Install the custom event filter
            getattr(self, box).wheelEvent = lambda event: None

        # Disable some fields
        self.set_disabled((
            'features_minus_tbutton', 'time_variable_cbox', 'save_pbutton'))

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

        # Reset the table selection when clicking outside
        self.features_table.setCurrentIndex(QModelIndex())
        self.features_table.clearSelection()

        # Disable the 'minus' button
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

        # Loop over the field names with 'currentTextChanged' events
        for key, value in {
                'column_cbox': self.update_save_button,
                'viewpoint_cbox': self.update_by_viewpoint,
                'time_variable_cbox': self.update_save_button,
                }.items():

            # Connect the 'currentTextChanged' event
            getattr(self, key).currentTextChanged.connect(value)

        # Connect the vertical table header with the row selection
        self.features_table.verticalHeader().sectionClicked.connect(
            self.select_row)

    def select_row(
            self,
            index):
        """
        Select a row in the table.

        Parameters
        ----------
        index : int
            Row index of the clicked vertical header.
        """

        # Set the selection mode to 'single selection'
        self.features_table.setSelectionMode(QAbstractItemView.SingleSelection)

        # Set the selection behavior
        self.features_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Select the row
        self.features_table.selectRow(index)

        # Reset the selection mode to 'no selection'
        self.features_table.setSelectionMode(QAbstractItemView.NoSelection)

        # Enable the 'minus' button
        self.features_minus_tbutton.setEnabled(True)

    def load(
            self,
            data_columns):
        """
        Load the data columns dictionary into the table.

        Parameters
        ----------
        data_columns : dict
            Dictionary with the column information on features and label.
        """

        # Load the data column names from the data
        column_names = self.load_names_from_data()

        # Set the initial number of rows and columns
        self.features_table.setRowCount(0)
        self.features_table.setColumnCount(5)

        # Add the horizontal header labels
        self.features_table.setHorizontalHeaderLabels(
            ['Scale', 'Segment', 'Function', 'Argument', 'Value'])

        # 
        self.features_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        # Add the dropdown menu to the features 'plus' button
        self.add_dropdown_to_features(column_names)

        # Get the features
        features = {
            key: value for key, value in data_columns.items()
            if value['type'] == 'feature'}

        # Get the label
        label = {
            key: value for key, value in data_columns.items()
            if value['type'] == 'label'}

        # 
        self.column_cbox.clear()
        self.time_variable_cbox.clear()

        # 
        self.column_cbox.addItems([''] + column_names)
        self.time_variable_cbox.addItems([''] + column_names)

        # 
        self.column_cbox.model().sort(0)
        self.time_variable_cbox.model().sort(0)

        # Set the feature values
        apply(self.set_feature, features.items())

        # Set the label values
        apply(self.set_label, label.items())

    def set_feature(
            self,
            item):
        """."""

        # 
        key, values = item

        # 
        nrow = self.features_table.rowCount()

        # 
        self.features_table.insertRow(nrow)

        # 
        header = QTableWidgetItem()
        header.setText(key)
        self.features_table.setVerticalHeaderItem(nrow, header)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems(['metric', 'nominal', 'ordinal'])
        widget.setCurrentText(values['scale'])
        self.features_table.setCellWidget(nrow, 0, widget)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems(
            ['', self.parent.segment_cbox.currentText()]
            + self.parent.segment_link_cbox.currentData())
        widget.adjustSize()
        widget.setCurrentText(
            '' if not values['segment'] else values['segment'])
        self.features_table.setCellWidget(nrow, 1, widget)

        # 
        widget = QComboBox()
        widget.wheelEvent = lambda event: None
        widget.addItems([''] + list(feature_map.keys()))
        widget.adjustSize()
        widget.setCurrentText(
            '' if not values['function'] else values['function'])
        widget.currentTextChanged.connect(
            partial(self.update_by_function, row=nrow))
        self.features_table.setCellWidget(nrow, 2, widget)

        # 
        self.update_by_function(nrow, values['argument'])

        # 
        widget = QLineEdit()
        widget.setText(values['value'])
        self.features_table.setCellWidget(nrow, 4, widget)

    def set_label(
            self,
            item):
        """."""

        # 
        key, values = item

        # 
        self.column_cbox.setCurrentText(key)

        # 
        self.viewpoint_cbox.setCurrentText(values['viewpoint'])

        # 
        self.time_variable_cbox.setCurrentText(
            '' if not values['time_variable'] else values['time_variable'])

        # 
        self.lower_bound_ledit.setText(
            '' if not values['bounds'] or values['bounds'][0] == 1.0
            else str(values['bounds'][0]))
        self.upper_bound_ledit.setText(
            '' if not values['bounds'] or values['bounds'][1] == 1.0
            else str(values['bounds'][1]))

    def load_names_from_data(self):
        """Load the column names from the data."""

        try:

            # Get the configuration file path
            configuration_path = ''.join(
                (self.parent.model_path_ledit.text(), '/configuration.json'))

            # Open a file stream
            with open(configuration_path, 'r', encoding='utf-8') as file:

                # Load the configuration
                configuration = load(file)

            # Get the column names from the configuration
            model_columns = (configuration['feature_names'] + list(filter(
                None, [configuration['label_name'],
                       configuration['time_variable_name']])))

        except Exception:

            # Set the model column names to empty
            model_columns = []

        try:

            # Get the column names from the dataset
            tab_data_columns = list(
                read_csv(self.parent.data_path_ledit.text()).columns)

        except Exception:

            # Set the dataset column names to empty
            tab_data_columns = []

        # Check if model column names (feature and label) have been passed
        if len(model_columns) >= 2:

            # Return the sorted model column names
            return sorted(model_columns)

        # Else, return the sorted tabular data column names
        return sorted(tab_data_columns)

    def add_dropdown_to_features(self, column_names):
        """Add the dropdown menu to the features 'plus' button."""

        # Reset the dropdown menu
        self.features_plus_tbutton.setMenu(None)

        # Initialize the dropdown menu
        menu = QMenu()

        # Loop over the column names
        for column in column_names:

            # Add the column to the dropdown menu
            menu.addAction(column, partial(self.add_default_feature, column))

        # Set the popup mode for the features 'plus' button
        self.features_plus_tbutton.setPopupMode(2)

        # Add the dropdown menu to the 'plus' button
        self.features_plus_tbutton.setMenu(menu)

    def add_default_feature(self, label):
        """."""

        # 
        self.set_feature((
            label,
            {'type': 'feature',
             'scale': 'metric',
             'segment': self.parent.segment_cbox.itemText(0),
             'function': next(iter(feature_map)),
             'argument': '',
             'value': ''}))

    def remove_feature(self):
        """Remove the selected feature."""

        # Remove the component item from the list widget
        self.features_table.removeRow(self.features_table.currentRow())

        # Disable some fields
        self.set_disabled(('features_minus_tbutton',))

    def update_by_function(
            self,
            row,
            value=None):
        """."""

        # 
        function = self.features_table.cellWidget(row, 2).currentText()

        # 
        if function in ('Dx', 'Vx'):

            # 
            widget = QSpinBox()
            widget.wheelEvent = lambda event: None
            widget.setRange(1, 99)
            widget.setValue(value if value else 1)

        # 
        elif function == 'Dose Gradient':

            # 
            widget = QComboBox()
            widget.wheelEvent = lambda event: None
            widget.addItems(['x', 'y', 'z'])
            widget.setCurrentText(value if value else 'x')

        # 
        elif function == 'Dose Moment':

            # 
            widget = QLineEdit()
            widget.setText(value if value else '')

        # 
        elif function == 'Dose Subvolume':

            # 
            widget = QComboBox()
            widget.wheelEvent = lambda event: None
            widget.addItems([
                'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3'])
            widget.setCurrentText(value if value else 'x1of2')

        else:

            # 
            widget = QLineEdit()
            widget.setText('')
            widget.setEnabled(False)

        # 
        self.features_table.setCellWidget(row, 3, widget)

    def update_by_viewpoint(self):
        """."""

        # 
        if self.viewpoint_cbox.currentText() in (
                'early', 'late', 'long-term', 'profile'):

            # 
            self.time_variable_cbox.setEnabled(True)

        else:

            # 
            self.time_variable_cbox.setEnabled(False)

            # 
            self.time_variable_cbox.setCurrentIndex(0)

    def save(self):
        """."""

        def read_cells(row):
            """."""

            # 
            scale = self.features_table.cellWidget(row, 0).currentText()

            # 
            segment = self.features_table.cellWidget(row, 1).currentText()

            # 
            function = self.features_table.cellWidget(row, 2).currentText()

            # 
            if function in ('Dx', 'Vx'):

                # 
                argument = self.features_table.cellWidget(row, 3).value()

            # 
            elif function in ('Dose Gradient', 'Dose Subvolume'):

                # 
                argument = self.features_table.cellWidget(row, 3).currentText()

            # 
            elif function == 'Dose Moment':

                # 
                argument = self.features_table.cellWidget(row, 3).text()

            else:

                # 
                argument = None

            try:

                # 
                value = string_to_numeric(
                    self.features_table.cellWidget(row, 4).text())

            except ValueError:

                value = (
                    None if self.features_table.cellWidget(row, 4).text() == ''
                    else self.features_table.cellWidget(row, 4).text())

            return {
                'type': 'feature', 'scale': scale, 'segment': segment,
                'function': function, 'argument': argument, 'value': value}

        # 
        features = {
            self.features_table.verticalHeaderItem(row).text(): read_cells(row)
            for row in range(self.features_table.rowCount())}

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

        # 
        self.parent.update_buttons()

        # 
        self.close()

    def update_save_button(self):
        """."""

        # 
        if (self.column_cbox.currentText() == '' or
            (self.viewpoint_cbox.currentText() != 'longitudinal' and
             self.time_variable_cbox.currentText() == '') or
                self.features_table.rowCount() == 0):

            # 
            self.save_pbutton.setEnabled(False)

        else:

            # 
            self.save_pbutton.setEnabled(True)

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
