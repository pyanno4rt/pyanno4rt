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
    cbox, ledit, pbutton_composer, sbox, tbutton_composer)
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.tools import apply, string_to_numeric

# %% Class definition


class DataColumnsWindow(QMainWindow, Ui_data_columns_window):
    """
    Data columns window for the machine learning model-based components.

    This class sets up the data columns window for the machine learning \
    model-based components in the graphical user interface, including the \
    feature table and label fields.
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

        # Initialize the model load indicator
        self.from_model = False

        # Set the stylesheets
        self.set_styles({
            'feature_plus_tbutton': tbutton_composer,
            'feature_minus_tbutton': tbutton_composer,
            'column_cbox': cbox,
            'viewpoint_cbox': cbox,
            'time_variable_cbox': cbox,
            'lower_bound_ledit': ledit,
            'upper_bound_ledit': ledit,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox elements
        for box in ('column_cbox', 'viewpoint_cbox', 'time_variable_cbox'):

            # Install the custom event filter
            getattr(self, box).wheelEvent = lambda event: None

        # Disable some fields
        self.set_disabled((
            'feature_minus_tbutton', 'time_variable_cbox', 'save_pbutton'))

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
        self.feature_table.setCurrentIndex(QModelIndex())
        self.feature_table.clearSelection()

        # Disable the 'minus' button
        self.feature_minus_tbutton.setEnabled(False)

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
                'feature_minus_tbutton': self.remove_feature,
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
        self.feature_table.verticalHeader().sectionClicked.connect(
            self.select_row)

    def select_row(
            self,
            index):
        """
        Select a row in the table.

        Parameters
        ----------
        index : int
            Row index of the selected vertical header.
        """

        # Set the selection mode to 'single selection'
        self.feature_table.setSelectionMode(QAbstractItemView.SingleSelection)

        # Set the selection behavior
        self.feature_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Select the row
        self.feature_table.selectRow(index)

        # Reset the selection mode to 'no selection'
        self.feature_table.setSelectionMode(QAbstractItemView.NoSelection)

        # Check if the columns are not loaded from a model folder
        if not self.from_model:

            # Enable the 'minus' button
            self.feature_minus_tbutton.setEnabled(True)

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

        # Set the initial number of rows and columns
        self.feature_table.setRowCount(0)
        self.feature_table.setColumnCount(5)

        # Add the horizontal header labels
        self.feature_table.setHorizontalHeaderLabels(
            ['Scale', 'Segment', 'Function', 'Argument', 'Value'])

        # Set the resize mode for the horizontal section
        self.feature_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        # Load the data column names from the data
        column_names = self.load_names_from_data()

        # Check if the columns are not loaded from a model folder
        if not self.from_model:

            # Add the dropdown menu to the 'plus' button
            self.add_dropdown_to_features(column_names)

            # Enable the 'plus' button
            self.feature_plus_tbutton.setEnabled(True)

        else:

            # Disable the 'plus' button
            self.feature_plus_tbutton.setEnabled(False)

        # Clear the variable label combo boxes
        self.column_cbox.clear()
        self.time_variable_cbox.clear()

        # Add the column names to the label combo boxes
        self.column_cbox.addItems([''] + column_names)
        self.time_variable_cbox.addItems([''] + column_names)

        # Insert the feature values
        apply(self.insert_feature, {
            key: value for key, value in data_columns.items()
            if value['type'] == 'feature'}.items())

        # Insert the label values
        apply(self.insert_label, {
            key: value for key, value in data_columns.items()
            if value['type'] == 'label'}.items())

        # Update the time variable combo box
        self.update_by_viewpoint()

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

        except FileNotFoundError:

            # Set the model column names to the empty list
            model_columns = []

        try:

            # Get the column names from the dataset
            tab_data_columns = list(
                read_csv(self.parent.data_path_ledit.text()).columns)

        except FileNotFoundError:

            # Set the dataset column names to the empty list
            tab_data_columns = []

        # Check if model column names have been passed
        if len(model_columns) >= 2 and not len(tab_data_columns) >= 2:

            # Set the model load indicator to True
            self.from_model = True

            # Return the sorted model column names
            return model_columns

        # Set the model load indicator to False
        self.from_model = False

        # Return the sorted tabular data column names
        return tab_data_columns

    def add_dropdown_to_features(
            self,
            column_names):
        """
        Add the dropdown menu to the feature 'plus' button.

        Parameters
        ----------
        column_names : list
            The column names read from the model folder or data file path.
        """

        # Reset the dropdown menu
        self.feature_plus_tbutton.setMenu(None)

        # Initialize the dropdown menu
        menu = QMenu()

        # Check if the number of columns is greater than 25
        if len(column_names) > 25:

            # Group the column names into chunks of max. 25 elements
            chunks = [
                column_names[i:i+25] for i in range(0, len(column_names), 25)]

            # Loop over the chunks
            for i, chunk in enumerate(chunks):

                # Add a submenu for the chunk
                submenu = menu.addMenu(f'Columns {25*i+1}-{25*(i+1)}')

                # Loop over the column names in the chunk
                for column in chunk:

                    # Add the action to the submenu
                    submenu.addAction(
                        column, partial(self.add_default_feature, column))

        else:

            # Loop over the column names
            for column in column_names:

                # Add the column to the dropdown menu
                menu.addAction(
                    column, partial(self.add_default_feature, column))

        # Set the popup mode for the 'plus' button
        self.feature_plus_tbutton.setPopupMode(2)

        # Add the dropdown menu to the 'plus' button
        self.feature_plus_tbutton.setMenu(menu)

    def add_default_feature(
            self,
            label):
        """
        Add a new (default) row to the feature table.

        Parameters
        ----------
        label : str
            Name of the feature to be added.
        """

        # Insert the feature
        self.insert_feature((label, {
            'type': 'feature', 'scale': 'metric', 'segment': '',
            'function': '', 'argument': '', 'value': ''}))

        # Move the feature table slider to the bottom
        self.feature_table.verticalScrollBar().setSliderPosition(
            self.feature_table.verticalScrollBar().maximum())

        # Update the save button
        self.update_save_button()

    def insert_feature(
            self,
            item):
        """
        Insert the feature information.

        Parameters
        ----------
        item : tuple
            Tuple with the feature label and parameter dictionary.
        """

        def add_combo_box(items, current_text, row, column, action=None):
            """Add a combo box with given properties to the table."""

            # Initialize the combo box
            combo_box = QComboBox()

            # Filter the wheel event
            combo_box.wheelEvent = lambda event: None

            # Set the stylesheet
            combo_box.setStyleSheet(cbox)

            # Add the items
            combo_box.addItems(items)

            # Adjust the size
            combo_box.adjustSize()

            # Set the current text
            combo_box.setCurrentText(current_text)

            # Check if the columns are loaded from a model folder
            if self.from_model:

                # Disable the combo box
                combo_box.setEnabled(False)

            # Check if an action has been passed
            if action is not None:

                # Connect the action with the signal
                combo_box.currentTextChanged.connect(action)

            # Set the cell widget in the feature table
            self.feature_table.setCellWidget(row, column, combo_box)

        # Get the label and the parameters
        label, parameters = item

        # Get the index as the current row count
        index = self.feature_table.rowCount()

        # Insert a row at the index
        self.feature_table.insertRow(index)

        # Initialize the vertical header
        header = QTableWidgetItem()

        # Set the header text to the label
        header.setText(label)

        # Set the vertical header in the feature table
        self.feature_table.setVerticalHeaderItem(index, header)

        # Add the combo box for the feature scale
        add_combo_box(
            items=['metric', 'nominal', 'ordinal'],
            current_text=parameters['scale'],
            row=index,
            column=0)

        # Add the combo box for the feature segment
        add_combo_box(
            items=(['', self.parent.segment_cbox.currentText()]
                   + self.parent.segment_link_cbox.currentData()),
            current_text=(
                '' if not parameters['segment'] else parameters['segment']),
            row=index,
            column=1)

        # Add the combo box for the feature function
        add_combo_box(
            items=[''] + list(feature_map.keys()),
            current_text=(
                '' if not parameters['function'] else parameters['function']),
            row=index,
            column=2,
            action=partial(self.update_by_function, index=index))

        # Add the feature argument field depending on the function
        self.update_by_function(index, parameters['argument'])

        # Initialize a line edit for the feature value
        line_edit = QLineEdit()

        # Set the stylesheet
        line_edit.setStyleSheet(ledit)

        # Set the text to the current value
        line_edit.setText(parameters['value'])

        # Check if the columns are loaded from a model folder
        if self.from_model:

            # Disable the line edit
            line_edit.setEnabled(False)

        # Set the cell widget in the feature table
        self.feature_table.setCellWidget(index, 4, line_edit)

    def insert_label(
            self,
            item):
        """
        Insert the label information.

        Parameters
        ----------
        item : tuple
            Tuple with the label name and parameter dictionary.
        """

        # Get the name and the parameters
        label, parameters = item

        # Set the label name
        self.column_cbox.setCurrentText(label)

        # Set the label viewpoint
        self.viewpoint_cbox.setCurrentText(parameters['viewpoint'])

        # Set the time variable
        self.time_variable_cbox.setCurrentText(
            '' if not parameters['time_variable']
            else parameters['time_variable'])

        # Loop over the lower and upper label bound fields
        for i, field in enumerate(('lower_bound_ledit', 'upper_bound_ledit')):

            # Set the field text
            getattr(self, field).setText(
                ''
                if not parameters['bounds'] or parameters['bounds'][i] == 1.0
                else str(parameters['bounds'][i]))

        # Check if the columns are loaded from a model folder
        if self.from_model:

            # Disable some fields
            self.set_disabled((
                'column_cbox', 'viewpoint_cbox', 'time_variable_cbox',
                'lower_bound_ledit', 'upper_bound_ledit'))

        else:

            # Enable some fields
            self.set_enabled((
                'column_cbox', 'viewpoint_cbox', 'time_variable_cbox',
                'lower_bound_ledit', 'upper_bound_ledit'))

    def remove_feature(self):
        """Remove the selected feature."""

        # Remove the component item from the list widget
        self.feature_table.removeRow(self.feature_table.currentRow())

        # Disable the 'minus' button
        self.feature_minus_tbutton.setEnabled(False)

        # Update the save button
        self.update_save_button()

    def update_by_function(
            self,
            index,
            value=None):
        """
        Update the argument in the feature table by the function.

        Parameters
        ----------
        index : int
            Row index of the argument field to be updated.

        value : int or str, default=None
            Value of the argument.
        """

        # Get the feature function
        function = self.feature_table.cellWidget(index, 2).currentText()

        # Check if the function is 'Dx' or 'Vx'
        if function in ('Dx', 'Vx'):

            # Initialize the spin box
            widget = QSpinBox()

            # Filter the wheel event
            widget.wheelEvent = lambda event: None

            # Set the stylesheet
            widget.setStyleSheet(sbox)

            # Set the button symbol to plus/minus
            widget.setButtonSymbols(1)

            # Set the value range
            widget.setRange(1, 99)

            # Set the initial value
            widget.setValue(value if value else 1)

        # Else, check if the function is 'Dose Gradient'
        elif function == 'Dose Gradient':

            # Initialize the combo box
            widget = QComboBox()

            # Filter the wheel event
            widget.wheelEvent = lambda event: None

            # Set the stylesheet
            widget.setStyleSheet(cbox)

            # Add the items
            widget.addItems(['x', 'y', 'z'])

            # Set the initial text
            widget.setCurrentText(value if value else 'x')

        # Else, check if the function is 'Dose Moment'
        elif function == 'Dose Moment':

            # Initialize the line edit
            widget = QLineEdit()

            # Set the stylesheet
            widget.setStyleSheet(ledit)

            # Set the initial text
            widget.setText(value if value else '')

        # Else, check if the function is 'Dose Subvolume'
        elif function == 'Dose Subvolume':

            # Initialize the combo box
            widget = QComboBox()

            # Filter the wheel event
            widget.wheelEvent = lambda event: None

            # Set the stylesheet
            widget.setStyleSheet(cbox)

            # Add the items
            widget.addItems([
                'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3',
                'y1of2', 'y2of2', 'y1of3', 'y2of3', 'y3of3'])

            # Set the initial text
            widget.setCurrentText(value if value else 'x1of2')

        else:

            # Initialize the line edit
            widget = QLineEdit()

            # Set the stylesheet
            widget.setStyleSheet(ledit)

            # Set the default empty text
            widget.setText('')

            # Disable the line edit
            widget.setEnabled(False)

        # Check if the columns are loaded from a model folder
        if self.from_model:

            # Disable the widget
            widget.setEnabled(False)

        # Set the cell widget in the feature table
        self.feature_table.setCellWidget(index, 3, widget)

    def update_by_viewpoint(self):
        """Update the time variable by the viewpoint."""

        # Check if the viewpoint requires a time variable
        if self.viewpoint_cbox.currentText() in (
                'early', 'late', 'long-term', 'profile'):

            # Enable the time variable combo box
            self.time_variable_cbox.setEnabled(True)

        else:

            # Disable the time variable combo box
            self.time_variable_cbox.setEnabled(False)

            # Reset the current index
            self.time_variable_cbox.setCurrentIndex(0)

    def save(self):
        """Save the data columns."""

        def read_cells(index):
            """Read the cells for a row index."""

            # Get the feature scale
            scale = self.feature_table.cellWidget(index, 0).currentText()

            # Get the feature segment
            segment = self.feature_table.cellWidget(index, 1).currentText()

            # Get the feature function
            function = self.feature_table.cellWidget(index, 2).currentText()

            # Check if the function is 'Dx' or 'Vx'
            if function in ('Dx', 'Vx'):

                # Get the argument from the spin box
                argument = self.feature_table.cellWidget(index, 3).value()

            # Else, check if the feature is 'Dose Gradient' or 'Dose Subvolume'
            elif function in ('Dose Gradient', 'Dose Subvolume'):

                # Get the argument from the combo box
                argument = (
                    self.feature_table.cellWidget(index, 3).currentText())

            # Else, check if the feature is 'Dose Moment'
            elif function == 'Dose Moment':

                # Get the argument from the line edit
                argument = self.feature_table.cellWidget(index, 3).text()

            else:

                # Set the argument to None
                argument = None

            try:

                # Convert the value from string to numeric
                value = string_to_numeric(
                    self.feature_table.cellWidget(index, 4).text())

            except ValueError:

                # Get the value from the default or as string
                value = (
                    None
                    if self.feature_table.cellWidget(index, 4).text() == ''
                    else self.feature_table.cellWidget(index, 4).text())

            return {
                'type': 'feature', 'scale': scale, 'segment': segment,
                'function': function, 'argument': argument, 'value': value}

        # Set up the feature dictionary
        features = {
            self.feature_table.verticalHeaderItem(row).text(): read_cells(row)
            for row in range(self.feature_table.rowCount())}

        # Set up the label dictionary
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

        # Overwrite the parent data columns dictionary
        self.parent.data_columns = features | label

        # Update the parent button status
        self.parent.update_buttons()

        # Close the window
        self.close()

    def update_save_button(self):
        """Update the save button by the conditions."""

        # Check if any condition blocks the save button
        if (self.column_cbox.currentText() == '' or
            (self.viewpoint_cbox.currentText() != 'longitudinal' and
             self.time_variable_cbox.currentText() == '') or
                self.feature_table.rowCount() == 0):

            # Disable the save button
            self.save_pbutton.setEnabled(False)

        else:

            # Enable the save button
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
