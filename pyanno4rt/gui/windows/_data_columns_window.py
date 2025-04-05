"""Data columns window."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from json import load
from pandas import read_csv
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QInputDialog, QLineEdit,
    QMainWindow, QMenu, QSpinBox, QTableWidgetItem)

# %% Internal package import

from pyanno4rt.gui.compilations.data_columns_window import (
    Ui_data_columns_window)
from pyanno4rt.gui.styles._custom_styles import (
    cbox, ledit, pbutton_composer, sbox, tbutton_composer)
from pyanno4rt.learning.features import DynamicFeature, Label, StaticFeature
import pyanno4rt.learning._maps as maps
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
            'preset_load_tbutton': tbutton_composer,
            'preset_save_tbutton': tbutton_composer,
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
                'preset_load_tbutton': self.load_preset,
                'preset_save_tbutton': self.save_preset,
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

    def save_preset(self):
        """Save the field inputs to a preset."""

        # Get the text input and state from the input dialog
        text, checked = QInputDialog.getText(
            self, "Save preset", "Enter preset name:", QLineEdit.Normal, "")

        # Check if the dialog has been confirmed with a non-empty text
        if checked and text != '':

            # Save the data columns to the presets dictionary
            self.parent.parent.data_presets[text] = self.read_columns()

    def load_preset(self):
        """Load the field inputs from a preset."""

        # Get the selected preset and state from the input dialog
        selection, checked = QInputDialog.getItem(
            self, "Load preset", "Select preset:",
            [''] + list(self.parent.parent.data_presets),
            current=0, editable=False)

        # Check if the dialog has been confirmed with a non-empty selection
        if checked and selection != '':

            # Load the data columns from the presets dictionary
            self.load(self.parent.parent.data_presets.get(selection, []))

    def load(self, preset=None):
        """Load the data columns into the table."""

        # Set the initial number of rows and columns
        self.feature_table.setRowCount(0)
        self.feature_table.setColumnCount(5)

        # Add the horizontal header labels
        self.feature_table.setHorizontalHeaderLabels(
            ['Segment', 'Function', 'Argument', 'Value', 'Scale'])

        # Set the resize mode for the horizontal section
        self.feature_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        # Check if no preset has been passed
        if preset is None:

            # Load the column names and list from the dataset/model
            column_names, data_columns = self.load_columns_from_data()

        else:

            # Load the column names from the dataset
            column_names, _ = self.load_columns_from_data()

            # Load the data columns list from the preset
            data_columns = preset

        # Check if the columns are not loaded from a model folder
        if not self.from_model:

            # Add the dropdown menu to the 'plus' button
            self.add_dropdown_to_features(column_names)

            # Enable some fields
            self.set_enabled(('preset_load_tbutton', 'feature_plus_tbutton'))

        else:

            # Disable some fields
            self.set_disabled(('preset_load_tbutton', 'feature_plus_tbutton'))

        # Clear the variable label combo boxes
        self.column_cbox.clear()
        self.time_variable_cbox.clear()

        # Add the column names to the label combo boxes
        self.column_cbox.addItems([''] + column_names)
        self.time_variable_cbox.addItems([''] + column_names)

        # Insert the feature values
        apply(
            self.insert_feature,
            (next(iter(column.to_dict().values())) for column in data_columns
             if 'Feature' in type(column).__name__))

        # Insert the label values
        apply(
            self.insert_label,
            (next(iter(column.to_dict().values())) for column in data_columns
             if type(column).__name__ == 'Label'))

        # Update the time variable combo box
        self.update_by_viewpoint()

    def load_columns_from_data(self):
        """
        Load the data columns from the data.

        Returns
        -------
        list
            The column names read from the model folder or data file path.

        list
            List with the features and the label.
        """

        try:

            # Get the configuration file path
            configuration_path = ''.join(
                (self.parent.model_path_ledit.text(), '/configuration.json'))

            # Open a file stream
            with open(configuration_path, 'r', encoding='utf-8') as file:

                # Load the configuration
                configuration = load(file)

            # Get the column names from the configuration
            column_names = (configuration['feature_names'] + list(set(filter(
                None, [configuration['time_variable_name'],
                       configuration['label_name']]))))

            # Get the column objects from the configuration
            column_objects = (
                [DynamicFeature(
                    column=item[0],
                    segment=item[1]['segment'],
                    function=item[1]['function'],
                    argument=item[1]['argument'],
                    scale=configuration['feature_scales'][index])
                 if item[1]['value'] is None else
                 StaticFeature(
                     column=item[0],
                     value=item[1]['value'],
                     scale=configuration['feature_scales'][index])
                 for index, item in enumerate(
                     configuration['feature_definitions'].items())]
                + [Label(
                    column=configuration['label_name'],
                    viewpoint=configuration['label_viewpoint'],
                    time_variable=configuration['time_variable_name'],
                    bounds=configuration['label_bounds'])])

            # Loop over the column objects
            for item in column_objects:

                # Find the column-matching parent item
                parent_item = next((
                    element for element in self.parent.data_columns
                    if element.column == item.column), None)

                # Check if the parent item exists
                if parent_item is not None:

                    # Overwrite the item segment
                    item.segment = parent_item.segment

        except FileNotFoundError:

            # Set the column names as empty
            column_names = []

            # Set the columns objects as empty
            column_objects = []

        try:

            # Get the tabular column names from the dataset
            tab_data_columns = list(
                read_csv(self.parent.data_path_ledit.text()).columns)

        except FileNotFoundError:

            # Set the tabular column names as empty
            tab_data_columns = []

        # Set the column names and objects to the tabular data inputs
        columns, objects = tab_data_columns, self.parent.data_columns

        # Set the model load indicator to the default
        self.from_model = False

        # Check if column names have been passed
        if len(column_names) >= 2:

            # Set the model load indicator
            self.from_model = len(tab_data_columns) < 2

            # Check if the columns are loaded from a model folder
            if self.from_model:

                # Overwrite the column names by the model inputs
                columns = column_names

            # Overwrite the columns objects by the model inputs
            objects = column_objects

        # Return the column names and objects
        return columns, objects

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
                        column, partial(self.insert_default_feature, column))

        else:

            # Loop over the column names
            for column in column_names:

                # Add the column to the dropdown menu
                menu.addAction(
                    column, partial(self.insert_default_feature, column))

        # Set the popup mode for the 'plus' button
        self.feature_plus_tbutton.setPopupMode(2)

        # Add the dropdown menu to the 'plus' button
        self.feature_plus_tbutton.setMenu(menu)

    def insert_default_feature(
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
        self.insert_feature({
            'column': label, 'segment': '', 'function': '', 'argument': '',
            'value': '', 'scale': 'metric'})

        # Move the feature table slider to the bottom
        self.feature_table.verticalScrollBar().setSliderPosition(
            self.feature_table.verticalScrollBar().maximum())

        # Update the save button
        self.update_save_button()

    def insert_feature(
            self,
            feature):
        """
        Insert the feature information.

        Parameters
        ----------
        feature : dict
            Dictionary with information on the feature.
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
            if self.from_model and column != 1:

                # Disable the combo box
                combo_box.setEnabled(False)

            # Check if an action has been passed
            if action is not None:

                # Connect the action with the signal
                combo_box.currentTextChanged.connect(action)

            # Set the cell widget in the feature table
            self.feature_table.setCellWidget(row, column, combo_box)

        # Get the index as the current row count
        index = self.feature_table.rowCount()

        # Insert a row at the index
        self.feature_table.insertRow(index)

        # Initialize the vertical header
        header = QTableWidgetItem()

        # Set the header text to the column name
        header.setText(feature['column'])

        # Set the vertical header in the feature table
        self.feature_table.setVerticalHeaderItem(index, header)

        # Get the feature segment
        segment = feature.get('segment')

        # Add the combo box for the feature segment
        add_combo_box(
            items=(
                ['', self.parent.segment_cbox.currentText()]
                + self.parent.segment_link_cbox.currentData()),
            current_text=('' if segment is None else segment),
            row=index,
            column=0)

        # Get the feature function
        function = feature.get('function')

        # Add the combo box for the feature function
        add_combo_box(
            items=[''] + list(maps.FEATURES),
            current_text=('' if function is None else function),
            row=index,
            column=1,
            action=partial(self.update_by_function, index=index))

        # Add the feature argument field depending on the function
        self.update_by_function(index, feature.get('argument'))

        # Get the feature value
        value = feature.get('value')

        # Initialize a line edit for the feature value
        line_edit = QLineEdit()

        # Set the stylesheet
        line_edit.setStyleSheet(ledit)

        # Set the text to the current value
        line_edit.setText('' if value is None else str(value))

        # Check if the columns are loaded from a model folder
        if self.from_model:

            # Disable the line edit
            line_edit.setEnabled(False)

        # Set the cell widget in the feature table
        self.feature_table.setCellWidget(index, 3, line_edit)

        # Add the combo box for the feature scale
        add_combo_box(
            items=['metric', 'nominal', 'ordinal'],
            current_text=feature.get('scale'),
            row=index,
            column=4)

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
        function = self.feature_table.cellWidget(index, 1).currentText()

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
            widget.setValue(1 if value is None else value)

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
            widget.setCurrentText('x' if value is None else value)

        # Else, check if the function is 'Dose Moment'
        elif function == 'Dose Moment':

            # Initialize the line edit
            widget = QLineEdit()

            # Set the stylesheet
            widget.setStyleSheet(ledit)

            # Set the initial text
            widget.setText('' if value is None else value)

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
            widget.setCurrentText('x1of2' if value is None else value)

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
        self.feature_table.setCellWidget(index, 2, widget)

    def insert_label(
            self,
            label):
        """
        Insert the label information.

        Parameters
        ----------
        label : object of class \
            :class:`~pyanno4rt.learning.features._columns.Label`
            The object used to represent the label.
        """

        # Set the label name
        self.column_cbox.setCurrentText(label['column'])

        # Set the label viewpoint
        self.viewpoint_cbox.setCurrentText(label['viewpoint'])

        # Set the time variable
        self.time_variable_cbox.setCurrentText(
            '' if label['time_variable'] is None else label['time_variable'])

        # Loop over the lower and upper label bound fields
        for i, field in enumerate(('lower_bound_ledit', 'upper_bound_ledit')):

            # Set the field text
            getattr(self, field).setText(
                '' if label['bounds'] is None or label['bounds'][i] == 1.0
                else str(label['bounds'][i]))

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

    def update_by_viewpoint(self):
        """Update the time variable by the viewpoint."""

        # Check if the viewpoint requires a time variable
        if (self.viewpoint_cbox.currentText() in (
                'early', 'late', 'long-term', 'profile')
                and not self.from_model):

            # Enable the time variable combo box
            self.time_variable_cbox.setEnabled(True)

        else:

            # Disable the time variable combo box
            self.time_variable_cbox.setEnabled(False)

            # Check if the columns are not loaded from a model folder
            if not self.from_model:

                # Reset the current index
                self.time_variable_cbox.setCurrentIndex(0)

    def read_columns(self):
        """
        Read the data columns from the input fields.

        Returns
        -------
        list
            List with the features and the label.
        """

        def get_feature(row):
            """Convert the row cells to a feature."""

            # Get the column name
            column = self.feature_table.verticalHeaderItem(row).text()

            # Get the feature segment
            segment = self.feature_table.cellWidget(row, 0).currentText()

            # Get the feature function
            function = self.feature_table.cellWidget(row, 1).currentText()

            # Check if the function is 'Dx' or 'Vx'
            if function in ('Dx', 'Vx'):

                # Get the argument from the spin box
                argument = self.feature_table.cellWidget(row, 2).value()

            # Else, check if the feature is 'Dose Gradient' or 'Dose Subvolume'
            elif function in ('Dose Gradient', 'Dose Subvolume'):

                # Get the argument from the combo box
                argument = (
                    self.feature_table.cellWidget(row, 2).currentText())

            # Else, check if the feature is 'Dose Moment'
            elif function == 'Dose Moment':

                # Get the argument from the line edit
                argument = self.feature_table.cellWidget(row, 2).text()

            else:

                # Set the argument to None
                argument = None

            try:

                # Convert the value from string to numeric
                value = string_to_numeric(
                    self.feature_table.cellWidget(row, 3).text())

            except ValueError:

                # Get the value from the default or as string
                value = (
                    None
                    if self.feature_table.cellWidget(row, 3).text() == ''
                    else self.feature_table.cellWidget(row, 3).text())

            # Get the feature scale
            scale = self.feature_table.cellWidget(row, 4).currentText()

            # Check if the value is None
            if value is None:

                # Return a dynamic feature
                return DynamicFeature(
                    column, segment, function, argument, scale)

            # Return a static feature
            return StaticFeature(column, value)

        # Set up the feature list
        features = [
            get_feature(row) for row in range(self.feature_table.rowCount())]

        # Set up the label
        label = Label(
            self.column_cbox.currentText(), self.viewpoint_cbox.currentText(),
            (None if self.time_variable_cbox.currentText() == ''
             else self.time_variable_cbox.currentText()),
            [1 if bound == '' else None if bound == 'None'
             else string_to_numeric(bound) for bound in (
                 self.lower_bound_ledit.text(),
                 self.upper_bound_ledit.text())])

        return features + [label]

    def save(self):
        """Save the data columns."""

        # Overwrite the parent data columns list
        self.parent.data_columns = self.read_columns()

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
        """Close the data columns window."""

        # Hide the window
        self.hide()
