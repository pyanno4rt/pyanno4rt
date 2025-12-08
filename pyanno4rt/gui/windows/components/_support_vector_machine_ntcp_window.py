"""Support vector machine NTCP component window."""

# Author: Tim Ortkamp

# %% External package import

from json import loads
from os.path import abspath
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMainWindow

# %% Internal package import

from pyanno4rt.gui._custom_styles import (
    cbox, ledit, pbutton_composer, sbox, tbutton_composer, tbutton_data_window)
from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.compilations.components.support_vector_machine_ntcp_window import (
    Ui_support_vector_machine_ntcp_window)
from pyanno4rt.gui.custom_widgets import CheckableComboBox
from pyanno4rt.gui.windows import DataColumnsWindow
import pyanno4rt.learning._maps as maps
from pyanno4rt.optimization.components import SupportVectorMachineOutcome
from pyanno4rt.tools import string_to_numeric

# %% Class definition


class SupportVectorMachineNTCPWindow(
        QMainWindow, Ui_support_vector_machine_ntcp_window):
    """
    Support vector machine NTCP component window for the GUI.

    This class sets up the support vector machine NTCP component window for \
    the graphical user interface, including input fields for parameterization.

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

        # Initialize the data columns list
        self.data_columns = []

        # Initialize the edit boolean
        self.edit = False

        # Initialize the data columns window
        self.data_columns_window = DataColumnsWindow(self)

        # Initialize the checkable combo boxes
        self.segment_link_cbox = CheckableComboBox()
        self.kernel_cbox = CheckableComboBox()
        self.class_weight_cbox = CheckableComboBox()
        self.graphs_cbox = CheckableComboBox()
        self.kpi_cbox = CheckableComboBox()

        # Loop over the checkable combo boxes with their parameters
        for box, parameters in {
                'segment_link_cbox': (
                    426, list(self.parent.segments), False,
                    'segment_link_layout'),
                'kernel_cbox': (
                    181, ['linear', 'poly', 'rbf', 'sigmoid'], True,
                    'kernel_layout'),
                'class_weight_cbox': (
                    181, ['None', 'balanced'], True, 'class_weight_layout'),
                'graphs_cbox': (
                    261, ['AUC-ROC', 'AUC-PR', 'F1'], True, 'graphs_layout'),
                'kpi_cbox': (
                    401, ['Logloss', 'Brier score', 'Subset accuracy',
                          'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                          'Precision', 'Recall', 'F1 score', 'MCC', 'AUC'],
                    True, 'kpi_layout')
                }.items():

            # Get the combo box
            combo_box = getattr(self, box)

            # Set a fixed size
            combo_box.setFixedSize(parameters[0], 31)

            # Add the items
            combo_box.addItems(parameters[1], parameters[2])

            # Add the combo box to the layout
            getattr(self, parameters[3]).addWidget(combo_box)

        # Add the segment items to the segment combo box
        self.segment_cbox.addItems(list(self.parent.segments))
        self.segment_cbox.setCurrentIndex(-1)

        # Add the losses to the tune score combo box
        self.tune_score_cbox.addItems(['AUC'] + list(maps.LOSSES))
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
            'model_path_ledit': ledit,
            'model_path_tbutton': tbutton_composer,
            'data_path_ledit': ledit,
            'data_path_tbutton': tbutton_composer,
            'data_columns_tbutton': tbutton_data_window,
            'prep_steps_ledit': ledit,
            'C_lower_bound_ledit': ledit,
            'C_upper_bound_ledit': ledit,
            'kernel_cbox': cbox,
            'gamma_lower_bound_ledit': ledit,
            'gamma_upper_bound_ledit': ledit,
            'degree_ledit': ledit,
            'tol_ledit': ledit,
            'class_weight_cbox': cbox,
            'tune_eval_sbox': sbox,
            'tune_score_cbox': cbox,
            'tune_splits_sbox': sbox,
            'tune_repeats_sbox': sbox,
            'oof_splits_sbox': sbox,
            'oof_repeats_sbox': sbox,
            'graphs_cbox': cbox,
            'kpi_cbox': cbox,
            'identifier_ledit': ledit,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox and QSpinBox elements
        for box in (
                'segment_cbox', 'type_cbox', 'embedding_cbox',
                'segment_link_cbox', 'rank_sbox', 'kernel_cbox',
                'class_weight_cbox', 'tune_eval_sbox', 'tune_score_cbox',
                'tune_splits_sbox', 'tune_repeats_sbox', 'oof_splits_sbox',
                'oof_repeats_sbox', 'graphs_cbox', 'kpi_cbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(parent)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'weight_ledit', 'lower_bound_ledit', 'upper_bound_ledit',
            'model_label_ledit', 'model_path_ledit', 'data_path_ledit',
            'prep_steps_ledit', 'C_lower_bound_ledit', 'C_upper_bound_ledit',
            'gamma_lower_bound_ledit', 'gamma_upper_bound_ledit',
            'degree_ledit', 'tol_ledit', 'identifier_ledit'))

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
        component : object of class \
            :class:`~pyanno4rt.optimization.components._support_vector_machine_ntcp.SupportVectorMachineNTCP`
            The object used to represent the optimization component.

        edit : bool
            Indicator for the editing of the component.
        """

        # Get the edit attribute from the argument
        self.edit = edit

        # Get the component parameters
        (segment, model_parameters, component_type, embedding, weight, rank,
         bounds, link, transform, identifier, display) = (
             component.arguments.values())

        # Convert the bounds
        lower, upper = (None, None) if bounds is None else bounds

        # Get the data columns dictionary
        self.data_columns = model_parameters.data_columns

        # Get the tune space
        tune_space = model_parameters.tune_space

        # Get the display options
        display_options = model_parameters.display_options

        # Loop over the fields with 'setText' method
        for key, value in {
                'weight_ledit': '' if weight == 1.0 else str(weight),
                'lower_bound_ledit': '' if lower is None else str(lower),
                'upper_bound_ledit': '' if upper is None else str(upper),
                'model_label_ledit': model_parameters.model_label,
                'model_path_ledit': (
                    '' if not model_parameters.model_folder_path
                    else abspath(model_parameters.model_folder_path)),
                'data_path_ledit': (
                    '' if not model_parameters.data_path
                    else abspath(model_parameters.data_path)),
                'prep_steps_ledit': (
                    '' if not model_parameters.preprocessing
                    or model_parameters.preprocessing == ['Identity']
                    else str(model_parameters.preprocessing).replace(
                        "\'", '')),
                'C_lower_bound_ledit': (
                    '' if tune_space.C[0] == 2**-5 else str(tune_space.C[0])),
                'C_upper_bound_ledit': (
                    '' if tune_space.C[1] == 2**10 else str(tune_space.C[1])),
                'degree_ledit': (
                    '' if tune_space.degree == [3, 4, 5, 6]
                    else str(tune_space.degree)),
                'gamma_lower_bound_ledit': (
                    '' if tune_space.gamma[0] == 2**-15
                    else str(tune_space.gamma[0])),
                'gamma_upper_bound_ledit': (
                    '' if tune_space.gamma[1] == 2**3
                    else str(tune_space.gamma[1])),
                'tol_ledit': (
                    '' if tune_space.tol == [1e-4, 1e-5, 1e-6]
                    else str(tune_space.tol)),
                'identifier_ledit': '' if not identifier else identifier
                }.items():

            # Set the text
            getattr(self, key).setText(value)

        # Loop over the fields with 'setCurrentText' method
        for key, value in {
                'segment_cbox': segment,
                'type_cbox': component_type,
                'embedding_cbox': embedding,
                'tune_score_cbox': model_parameters.tune_score
                }.items():

            # Set the text
            getattr(self, key).setCurrentText(value)

        # Loop over the fields with 'setValue' method
        for key, value in {
                'rank_sbox': rank,
                'tune_eval_sbox': model_parameters.tune_evaluations,
                'tune_splits_sbox': model_parameters.tune_splits,
                'tune_repeats_sbox': model_parameters.tune_repeats,
                'oof_splits_sbox': model_parameters.oof_splits,
                'oof_repeats_sbox': model_parameters.oof_repeats
                }.items():

            # Set the value
            getattr(self, key).setValue(value)

        # Loop over the fields with 'setCheckState' method
        for key, value in {
                'transform_check': 2*transform,
                'write_features_check': 2*model_parameters.write_features,
                'inspect_model_check': 2*model_parameters.inspect,
                'evaluate_model_check': 2*model_parameters.evaluate,
                'disp_component_check': 2*display
                }.items():

            # Set the check state
            getattr(self, key).setCheckState(value)

        # Loop over the checkable combo boxes with their selections
        for box, selection in {
                'segment_link_cbox': [] if not link else link,
                'kernel_cbox': tune_space.kernel,
                'class_weight_cbox': map(str, tune_space.class_weight),
                'graphs_cbox': display_options.graphs,
                'kpi_cbox': display_options.kpis
                }.items():

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
            'gamma_lower_bound_ledit', 'gamma_upper_bound_ledit',
            'degree_ledit', 'tol_ledit', 'identifier_ledit'))

    def save(self):
        """Save the fields to a component."""

        # Get the model parameters
        model_parameters = ModelParameters(
            model_label=self.model_label_ledit.text(),
            model_type='svm',
            model_folder_path=(
                None if self.model_path_ledit.text() == ''
                else abspath(self.model_path_ledit.text())),
            data_path=(
                None if self.data_path_ledit.text() == ''
                else abspath(self.data_path_ledit.text())),
            data_columns=self.data_columns,
            preprocessing=(
                ['Identity'] if self.prep_steps_ledit.text() == ''
                else self.prep_steps_ledit.text().strip('][').split(', ')),
            tune_space=TuneSpaceSVM(
                C=[
                    2**-5 if self.C_lower_bound_ledit.text() == ''
                    else string_to_numeric(self.C_lower_bound_ledit.text()),
                    2**10 if self.C_upper_bound_ledit.text() == ''
                    else string_to_numeric(self.C_upper_bound_ledit.text())],
                kernel=self.kernel_cbox.currentData(),
                degree=(
                    [3, 4, 5, 6] if self.degree_ledit.text() == ''
                    else loads(self.degree_ledit.text())),
                gamma=[
                    2**-15 if self.gamma_lower_bound_ledit.text() == '' else
                    string_to_numeric(self.gamma_lower_bound_ledit.text()),
                    2**3 if self.gamma_upper_bound_ledit.text() == '' else
                    string_to_numeric(self.gamma_upper_bound_ledit.text())],
                tol=(
                    [1e-4, 1e-5, 1e-6] if self.tol_ledit.text() == ''
                    else loads(self.tol_ledit.text())),
                class_weight=[
                    None if value == 'None' else value
                    for value in self.class_weight_cbox.currentData()]),
            tune_evaluations=self.tune_eval_sbox.value(),
            tune_score=self.tune_score_cbox.currentText(),
            tune_splits=self.tune_splits_sbox.value(),
            tune_repeats=self.tune_repeats_sbox.value(),
            inspect=self.inspect_model_check.isChecked(),
            evaluate=self.evaluate_model_check.isChecked(),
            oof_splits=self.oof_splits_sbox.value(),
            oof_repeats=self.oof_repeats_sbox.value(),
            write_features=self.write_features_check.isChecked(),
            display_options=DisplayOptions(
                graphs=self.graphs_cbox.currentData(),
                kpis=self.kpi_cbox.currentData()))

        # Get the component
        component = SupportVectorMachineOutcome(
            segment=self.segment_cbox.currentText(),
            model_parameters=model_parameters,
            component_type=self.type_cbox.currentText(),
            embedding=self.embedding_cbox.currentText(),
            weight=(
                1.0 if self.weight_ledit.text() == ''
                else string_to_numeric(self.weight_ledit.text())),
            rank=self.rank_sbox.value(),
            bounds=[
                None if self.lower_bound_ledit.text() == ''
                else string_to_numeric(self.lower_bound_ledit.text()),
                None if self.upper_bound_ledit.text() == ''
                else string_to_numeric(self.upper_bound_ledit.text())],
            link=(
                None if len(self.segment_link_cbox.currentData()) == 0
                else self.segment_link_cbox.currentData()),
            transform=self.transform_check.isChecked(),
            identifier=(
                None if self.identifier_ledit.text() == ''
                else self.identifier_ledit.text()),
            display=self.disp_component_check.isChecked())

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
            f'{component.arguments["component_type"]}_'
            f'{self.parent.segments[self.segment_cbox.currentText()]}']

        # Initialize the icon object
        icon = QIcon()

        # Add the pixmap to the icon
        icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

        # Get the component string
        component_string = ' - '.join((substring for substring in (
            self.segment_cbox.currentText(), component.name,
            f'weight: {component.arguments["weight"]}',
            f'embedding: {component.arguments["embedding"]}',
            f'link: {component.arguments["link"]}',
            f'identifier: {component.arguments["identifier"]}')
            if 'None' not in substring))

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
        self.data_columns_window.load()

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
        """Close the support vector machine NTCP window."""

        # Reset the edit boolean
        self.edit = False

        # Hide the window
        self.hide()
