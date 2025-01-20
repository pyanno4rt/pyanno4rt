"""Logistic regression NTCP component window."""

# Author: Tim Ortkamp

# %% External package import

from json import loads
from os.path import abspath
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.components.logistic_regression_ntcp_window import (
    Ui_logistic_regression_ntcp_window)
from pyanno4rt.gui.custom_widgets import CheckableComboBox
from pyanno4rt.gui.styles._custom_styles import (
    cbox, ledit, pbutton_composer, sbox, tbutton_composer, tbutton_data_window)
from pyanno4rt.gui.windows import DataColumnsWindow
from pyanno4rt.learning_model.losses import loss_map

# %% Class definition


class LogisticRegressionNTCPWindow(
        QMainWindow, Ui_logistic_regression_ntcp_window):
    """
    Logistic regression NTCP component window for the GUI.

    This class sets up the logistic regression NTCP component window for the \
    graphical user interface, including input fields for parameterization.

    Parameters
    ----------
    parent : object of class \
        :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, default=None
        The object representing the parent window for embedding.
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

        # Initialize the data columns dictionary
        self.data_columns = {}

        # Initialize the edit boolean
        self.edit = False

        # Initialize the data columns window
        self.data_columns_window = DataColumnsWindow(self)

        # Initialize the checkable combo boxes
        self.segment_link_cbox = CheckableComboBox()
        self.penalty_cbox = CheckableComboBox()
        self.class_weight_cbox = CheckableComboBox()
        self.graphs_cbox = CheckableComboBox()
        self.kpi_cbox = CheckableComboBox()

        # Loop over the checkable combo boxes with their parameters
        for box, parameters in {
                'segment_link_cbox': (
                    426, list(self.parent.segments.keys()), False,
                    'segment_link_layout'),
                'penalty_cbox': (
                    211, ['l1', 'l2', 'elasticnet'], True, 'penalty_layout'),
                'class_weight_cbox': (
                    211, ['None', 'balanced'], True, 'class_weight_layout'),
                'graphs_cbox': (
                    261, ['AUC-ROC', 'AUC-PR', 'F1'], True, 'graphs_layout'),
                'kpi_cbox': (
                    401, ['Logloss', 'Brier score', 'Subset accuracy',
                          'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                          'Precision', 'Recall', 'F1 score', 'MCC', 'AUC'],
                    True, 'kpi_layout')}.items():

            # Get the combo box
            combo_box = getattr(self, box)

            # Set a fixed size
            combo_box.setFixedSize(parameters[0], 31)

            # Add the items
            combo_box.addItems(parameters[1], parameters[2])

            # Add the combo box to the layout
            getattr(self, parameters[3]).addWidget(combo_box)

        # Add the segment items to the segment combo box
        self.segment_cbox.addItems(list(self.parent.segments.keys()))
        self.segment_cbox.setCurrentIndex(-1)

        # Add the losses to the tune score combo box
        self.tune_score_cbox.addItems(['AUC'] + list(loss_map.keys()))
        self.tune_score_cbox.model().sort(0)
        self.tune_score_cbox.setCurrentText('Logloss')

        # Set the stylesheets
        self.set_styles({
            'segment_cbox': cbox,
            'type_cbox': cbox,
            'embedding_cbox': cbox,
            'segment_link_cbox': cbox,
            'weight_ledit': ledit,
            'rank_sbox': sbox,
            'lower_bound_ledit': ledit,
            'upper_bound_ledit': ledit,
            'model_label_ledit': ledit,
            'model_path_tbutton': tbutton_composer,
            'data_path_tbutton': tbutton_composer,
            'data_columns_tbutton': tbutton_data_window,
            'prep_steps_ledit': ledit,
            'C_lower_bound_ledit': ledit,
            'C_upper_bound_ledit': ledit,
            'penalty_cbox': cbox,
            'tol_ledit': ledit,
            'class_weight_cbox': cbox,
            'tune_eval_sbox': sbox,
            'tune_score_cbox': cbox,
            'tune_splits_sbox': sbox,
            'oof_splits_sbox': sbox,
            'graphs_cbox': cbox,
            'kpi_cbox': cbox,
            'identifier_ledit': ledit,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox and QSpinBox elements
        for box in (
                'segment_cbox', 'segment_link_cbox', 'type_cbox',
                'embedding_cbox', 'rank_sbox', 'penalty_cbox',
                'class_weight_cbox', 'graphs_cbox', 'kpi_cbox',
                'tune_eval_sbox', 'tune_score_cbox', 'tune_splits_sbox',
                'oof_splits_sbox'):

            # Install the custom event filter
            getattr(self, box).installEventFilter(parent)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'weight_ledit', 'lower_bound_ledit', 'upper_bound_ledit',
            'model_label_ledit', 'model_path_ledit', 'data_path_ledit',
            'prep_steps_ledit', 'C_lower_bound_ledit', 'C_upper_bound_ledit',
            'tol_ledit', 'identifier_ledit'))

        # Disable some fields
        self.set_disabled(('data_columns_tbutton', 'save_pbutton'))

        # Connect the event signals
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
                'model_path_tbutton': self.add_model_path,
                'data_path_tbutton': self.add_data_path,
                'data_columns_tbutton': self.open_data_columns_window,
                'save_pbutton': self.save,
                'close_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Loop over the field names with 'currentTextChanged' events
        for key, value in {
                'segment_cbox': self.update_buttons
                }.items():

            # Connect the 'currentTextChanged' event
            getattr(self, key).currentTextChanged.connect(value)

        # Loop over the field names with 'textChanged' events
        for key, value in {
                'model_label_ledit': self.update_buttons,
                'model_path_ledit': self.update_buttons,
                'data_path_ledit': self.update_buttons
                }.items():

            # Connect the 'textChanged' event
            getattr(self, key).textChanged.connect(value)

    def load(
            self,
            component,
            edit=False):
        """
        Load a component into the window.

        Parameters
        ----------
        component : dict
            Dictionary with the information on the optimization component.

        edit : bool
            Indicator for the editing of the component.
        """

        # Get the edit attribute from the argument
        self.edit = edit

        # Get the segment associated with the component
        segment = next(iter(component))

        # Get the component parameters
        ctype = component[segment]['type']
        model_parameters = component[segment]['instance']['parameters'][
            'model_parameters']
        embedding = component[segment]['instance']['parameters'].get(
            'embedding', 'active')
        weight = component[segment]['instance']['parameters'].get(
            'weight', 1.0)
        rank = component[segment]['instance']['parameters'].get('rank', 1)
        lower, upper = component[segment]['instance']['parameters'].get(
            'bounds', (0.0, 1.0))
        link = component[segment]['instance']['parameters'].get('link', [])
        identifier = component[segment]['instance']['parameters'].get(
            'identifier')
        display = component[segment]['instance']['parameters'].get(
            'display', True)

        # Get the data columns dictionary
        self.data_columns = model_parameters['data_columns']

        # Get the tune space
        tune_space = model_parameters.get('tune_space', {
            'C': [2**-5, 2**10],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'tol': [1e-4, 1e-5, 1e-6],
            'class_weight': [None, 'balanced']})

        # Get the display options
        display_options = model_parameters.get(
            'display_options', {
                'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                'kpis': ['Logloss', 'Brier score', 'Subset accuracy',
                         'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                         'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']})

        # Loop over the fields with 'setText' method
        for key, value in {
                'weight_ledit': '' if weight == 1.0 else str(float(weight)),
                'lower_bound_ledit': '' if lower == 0.0 else str(float(lower)),
                'upper_bound_ledit': '' if upper == 1.0 else str(float(upper)),
                'model_label_ledit': model_parameters['model_label'],
                'model_path_ledit': (
                    '' if not model_parameters.get('model_folder_path')
                    else abspath(model_parameters['model_folder_path'])),
                'data_path_ledit': (
                    '' if not model_parameters.get('data_path')
                    else abspath(model_parameters['data_path'])),
                'prep_steps_ledit': (
                    '' if not model_parameters.get('preprocessing_steps')
                    or model_parameters['preprocessing_steps'] == ['Identity']
                    else str(model_parameters['preprocessing_steps']).replace(
                        "\'", '')),
                'C_lower_bound_ledit': (
                    '' if not tune_space.get('C')
                    or tune_space['C'][0] == 2**-5
                    else str(tune_space['C'][0])),
                'C_upper_bound_ledit': (
                    '' if not tune_space.get('C')
                    or tune_space['C'][1] == 2**10
                    else str(tune_space['C'][1])),
                'tol_ledit': (
                    '' if not tune_space.get('tol')
                    or tune_space['tol'] == [1e-4, 1e-5, 1e-6]
                    else str(tune_space['tol'])),
                'identifier_ledit': '' if not identifier else identifier
                }.items():

            # Set the field text
            getattr(self, key).setText(value)

        # Loop over the fields with 'setCurrentText' method
        for key, value in {
                'segment_cbox': segment,
                'type_cbox': ctype,
                'embedding_cbox': embedding,
                'tune_score_cbox': model_parameters.get(
                    'tune_score', 'Logloss')
                }.items():

            # Set the field text
            getattr(self, key).setCurrentText(value)

        # Loop over the fields with 'setValue' method
        for key, value in {
                'rank_sbox': rank,
                'tune_splits_sbox': model_parameters.get('tune_splits', 5),
                'oof_splits_sbox': model_parameters.get('oof_splits', 5),
                'tune_eval_sbox': model_parameters.get('tune_evaluations', 50)
                }.items():

            # Set the field text
            getattr(self, key).setValue(value)

        # Loop over the fields with 'setCheckState' method
        for key, value in {
                'write_features_check': (
                    2 if model_parameters.get('write_features', True) else 0),
                'inspect_model_check': (
                    2 if model_parameters.get('inspect_model', True) else 0),
                'evaluate_model_check': (
                    2 if model_parameters.get('evaluate_model', True) else 0),
                'disp_component_check': 2 if display else 0
                }.items():

            # Set the field text
            getattr(self, key).setCheckState(value)

        # Loop over the checkable combo boxes with their selections
        for box, selection in {
                'segment_link_cbox': link,
                'penalty_cbox': tune_space['penalty'],
                'class_weight_cbox': map(str, tune_space['class_weight']),
                'graphs_cbox': display_options['graphs'],
                'kpi_cbox': display_options['kpis']}.items():

            # Get the combo box
            combo_box = getattr(self, box)

            # Loop over the combo box items
            for item in (
                    combo_box.model().item(index)
                    for index in range(combo_box.count())):

                # Set the item text to checked or unchecked
                item.setCheckState(2*(item.text() in selection))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'weight_ledit', 'lower_bound_ledit', 'upper_bound_ledit',
            'model_label_ledit', 'model_path_ledit', 'data_path_ledit',
            'prep_steps_ledit', 'C_lower_bound_ledit', 'C_upper_bound_ledit',
            'tol_ledit', 'identifier_ledit'))

    def save(self):
        """Save the fields to a component."""

        # Get the model parameters
        model_parameters = {
            'model_label': self.model_label_ledit.text(),
            'model_folder_path': (
                None if self.model_path_ledit.text() == ''
                else abspath(self.model_path_ledit.text())),
            'data_path': (
                '' if self.data_path_ledit.text() == ''
                else abspath(self.data_path_ledit.text())),
            'data_columns': self.data_columns,
            'preprocessing_steps': (
                ['Identity'] if self.prep_steps_ledit.text() == ''
                else self.prep_steps_ledit.text().strip('][').split(', ')),
            'tune_space': {
                'C': (
                    [2**-5, 2**10] if any(bound == '' for bound in (
                        self.C_lower_bound_ledit.text(),
                        self.C_upper_bound_ledit.text()))
                    else [float(self.C_lower_bound_ledit.text()),
                          float(self.C_upper_bound_ledit.text())]),
                'penalty': self.penalty_cbox.currentData(),
                'tol': (
                    [1e-4, 1e-5, 1e-6] if self.tol_ledit.text() == ''
                    else loads(self.tol_ledit.text())),
                'class_weight': [
                    None if value == 'None' else value
                    for value in self.class_weight_cbox.currentData()]},
            'tune_evaluations': self.tune_eval_sbox.value(),
            'tune_score': self.tune_score_cbox.currentText(),
            'tune_splits': self.tune_splits_sbox.value(),
            'inspect_model': self.inspect_model_check.isChecked(),
            'evaluate_model': self.evaluate_model_check.isChecked(),
            'oof_splits': self.oof_splits_sbox.value(),
            'write_features': self.write_features_check.isChecked(),
            'display_options': {
                'graphs': self.graphs_cbox.currentData(),
                'kpis': self.kpi_cbox.currentData()}}

        # Configure the component dictionary
        component = {
            self.segment_cbox.currentText(): {
                'type': self.type_cbox.currentText(),
                'instance': {
                    'function': 'Logistic Regression NTCP',
                    'parameters': {
                        'model_parameters': model_parameters,
                        'embedding': self.embedding_cbox.currentText(),
                        'weight': (
                            1.0 if self.weight_ledit.text() == ''
                            else float(self.weight_ledit.text())),
                        'rank': self.rank_sbox.value(),
                        'bounds': [
                            0.0 if self.lower_bound_ledit.text() == ''
                            else float(self.lower_bound_ledit.text()),
                            1.0 if self.upper_bound_ledit.text() == ''
                            else float(self.upper_bound_ledit.text())],
                        'link': (
                            None
                            if len(self.segment_link_cbox.currentData()) == 0
                            else self.segment_link_cbox.currentData()),
                        'identifier': (
                            None if self.identifier_ledit.text() == ''
                            else self.identifier_ledit.text()),
                        'display': self.disp_component_check.isChecked()}}}}

        # Get the component and function parameters
        cparams = component[self.segment_cbox.currentText()]
        fparams = cparams['instance']['parameters']

        # Get the required parameter values
        ctype = cparams['type']
        identifier = fparams['identifier']
        embedding = f'embedding: {str(fparams["embedding"])}'
        weight = f'weight: {str(fparams["weight"])}'

        # Map the component and segment type to the icon paths
        paths = {
            'objective_TARGET': (
                ":/special_icons/icons_special/target-red-svgrepo-com.svg"),
            'objective_OAR': (
                ":/special_icons/icons_special/target-green-svgrepo-com.svg"),
            'constraint_TARGET': (
                ":/special_icons/icons_special/frame-red-svgrepo-com.svg"),
            'constraint_OAR': (
                ":/special_icons/icons_special/frame-green-svgrepo-com.svg")}

        # Get the icon path
        icon_path = paths[
            f'{ctype}_{self.parent.segments[self.segment_cbox.currentText()]}']

        # Initialize the icon object
        icon = QIcon()

        # Add the pixmap to the icon
        icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

        # Get the component string
        component_string = ' - '.join((substring for substring in (
            self.segment_cbox.currentText(), 'Logistic Regression NTCP',
            identifier, embedding, weight) if substring))

        # Check if the component item is edited
        if self.edit:

            # Remove the item from the plan components
            del self.parent.plan_components[self.parent.plan_ledit.text()][
                self.parent.components_lwidget.currentItem().text()]

            # Remove the item from the component list widget
            self.parent.components_lwidget.takeItem(
                self.parent.components_lwidget.currentRow())

            # Clear the selection in the component list widget
            self.parent.components_lwidget.selectionModel().clear()

            # Disable some fields
            self.parent.set_disabled(
                ('components_minus_tbutton', 'components_edit_tbutton'))

        # Add the component to the plan components
        self.parent.plan_components[self.parent.plan_ledit.text()][
            component_string] = component

        # Add the icon with the component string to the component list widget
        self.parent.components_lwidget.addItem(
            QListWidgetItem(icon, component_string))

        # Clear the selection in the component list widget
        self.parent.components_lwidget.selectionModel().clear()

        # Close the window
        self.close()

    def add_model_path(self):
        """Add the model path from a snapshot folder."""

        # Set the model folder path
        self.model_path_ledit.setText(QFileDialog.getExistingDirectory(
            self, 'Select a directory for loading'))

        # Set the model path field cursor position to zero
        self.model_path_ledit.setCursorPosition(0)

    def add_data_path(self):
        """Add the data path."""

        # Get the data file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an outcome data file', '', 'Outcome data (*.csv)')

        # Set the data path field
        self.data_path_ledit.setText(abspath(path))

        # Set the data path field cursor position to zero
        self.data_path_ledit.setCursorPosition(0)

    def open_data_columns_window(self):
        """Open the data columns window."""

        # Load the data columns into the window
        self.data_columns_window.load(self.data_columns)

        # Set the position of the window
        self.data_columns_window.position()

        # Show the window
        self.data_columns_window.show()

    def update_buttons(self):
        """Update the conditional buttons."""

        # Enable or disable the data columns button
        self.data_columns_tbutton.setEnabled(
            self.segment_cbox.currentText() != '' and
            any(text != '' for text in (
                self.model_path_ledit.text(), self.data_path_ledit.text())))

        # Enable or disable the save button
        self.save_pbutton.setEnabled(
            all(text != '' for text in (
                self.segment_cbox.currentText(),
                self.model_label_ledit.text()))
            and any(text != '' for text in (
                self.model_path_ledit.text(), self.data_path_ledit.text()))
            and len(self.data_columns) >= 2)

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the logistic regression NTCP window."""

        # Reset the edit boolean
        self.edit = False

        # Hide the window
        self.hide()
