"""Support vector machine TCP component window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import loads
from os.path import abspath, isfile
from pandas import read_csv
from PyQt5.QtCore import QDir, QEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QInputDialog, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QSpinBox)

# %% Internal package import

from pyanno4rt.gui.compilations.components.support_vector_machine_tcp_window import (
    Ui_support_vector_machine_tcp_window)
from pyanno4rt.gui.styles._custom_styles import (
    ledit, pbutton_composer, tbutton_composer)

# %% Class definition


class SupportVectorMachineTCPWindow(
        QMainWindow, Ui_support_vector_machine_tcp_window):
    """
    Support vector machine TCP component window for the application.

    This class creates a support vector machine TCP component window for the \
    graphical user interface, including input fields to parametrize the \
    component.
    """

    def __init__(
            self,
            parent=None):

        # Get the application from the argument
        self.parent = parent

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Loop over the QComboBox and QSpinBox elements
        for box in ('type_cbox', 'embedding_cbox', 'rank_sbox',
                    'filter_mode_cbox', 'label_viewpoint_cbox',
                    'tune_eval_sbox', 'tune_score_cbox', 'tune_splits_sbox',
                    'oof_splits_sbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(self)

        # 
        self.set_disabled(('features_minus_tbutton', 'features_load_tbutton',
                           'save_component_pbutton'))

        # 
        self.set_styles({'segment_ledit': ledit,
                         'link_ledit': ledit,
                         'identifier_ledit': ledit,
                         'model_label_ledit': ledit,
                         'model_path_tbutton': tbutton_composer,
                         'data_path_tbutton': tbutton_composer,
                         'features_plus_tbutton': tbutton_composer,
                         'features_minus_tbutton': tbutton_composer,
                         'features_load_tbutton': tbutton_composer,
                         'save_component_pbutton': pbutton_composer,
                         'close_component_pbutton': pbutton_composer})

        # 
        self.kernel_lwidget.addItems(['linear', 'rbf', 'poly', 'sigmoid'])
        self.class_weight_lwidget.addItems(['None', 'balanced'])
        self.graphs_lwidget.addItems(['AUC-ROC', 'AUC-PR', 'F1'])
        self.kpi_lwidget.addItems(['Logloss', 'Brier score', 'Subset accuracy',
                                   'Cohen Kappa', 'Hamming loss',
                                   'Jaccard score', 'Precision', 'Recall',
                                   'F1 score', 'MCC', 'AUC'])

        # 
        for index in range(self.kernel_lwidget.count()):
            self.kernel_lwidget.item(index).setCheckState(2)
        for index in range(self.class_weight_lwidget.count()):
            self.class_weight_lwidget.item(index).setCheckState(2)
        for index in range(self.graphs_lwidget.count()):
            self.graphs_lwidget.item(index).setCheckState(2)
        for index in range(self.kpi_lwidget.count()):
            self.kpi_lwidget.item(index).setCheckState(2)

        # 
        self.features_lwidget.setSpacing(4)
        self.kernel_lwidget.setSpacing(4)
        self.class_weight_lwidget.setSpacing(4)
        self.graphs_lwidget.setSpacing(4)
        self.kpi_lwidget.setSpacing(4)

        # 
        self.segment_ledit.textChanged.connect(self.update_save_button)

        # 
        self.model_path_ledit.textChanged.connect(self.update_save_button)
        self.model_path_tbutton.clicked.connect(self.add_model_path)

        # 
        self.data_path_ledit.textChanged.connect(self.update_save_button)
        self.data_path_ledit.textChanged.connect(self.update_by_data_path)
        self.data_path_tbutton.clicked.connect(self.add_data_path)

        # 
        self.features_plus_tbutton.clicked.connect(self.open_input_dialog)

        # 
        self.features_minus_tbutton.clicked.connect(self.remove_feature)

        # 
        self.features_lwidget.currentItemChanged.connect(
            lambda: self.set_enabled(('features_minus_tbutton',)))

        # 
        self.features_load_tbutton.clicked.connect(self.open_question_dialog)

        # 
        self.label_name_ledit.textChanged.connect(self.update_save_button)

        # 
        self.save_component_pbutton.clicked.connect(self.save)

        # 
        self.close_component_pbutton.clicked.connect(self.close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def load(self, component):
        """."""

        # Get the component parameters
        key = list(component.keys())[0]
        ctype = component[key]['type']
        model_params = component[key]['instance']['parameters'][
            'model_parameters']
        embedding = component[key]['instance']['parameters'].get(
            'embedding', 'active')
        weight = component[key]['instance']['parameters'].get('weight', 1)
        rank = component[key]['instance']['parameters'].get('rank', 1)
        lower, upper = component[key]['instance']['parameters'].get(
            'bounds', (0.0, 1.0))
        link = component[key]['instance']['parameters'].get('link')
        identifier = component[key]['instance']['parameters'].get('identifier')
        display = component[key]['instance']['parameters'].get('display', True)

        # Base
        self.segment_ledit.setText(key)
        self.type_cbox.setCurrentText(ctype)
        self.embedding_cbox.setCurrentText(embedding)

        # Function
        self.link_ledit.setText(
            '' if not link else str(link).replace("\'", ''))

        # Optimization
        self.weight_ledit.setText(
            '' if weight == 1 else str(float(weight)))
        self.rank_sbox.setValue(rank)
        self.lower_bound_ledit.setText(
            '' if lower == 0.0 else str(float(lower)))
        self.upper_bound_ledit.setText(
            '' if upper == 1.0 else str(float(upper)))

        # Outcome model - BASE
        self.model_label_ledit.setText(model_params['model_label'])
        self.model_path_ledit.setText(
            '' if not model_params.get('model_folder_path')
            else abspath(model_params['model_folder_path']))
        self.data_path_ledit.setText(
            '' if not model_params.get('data_path')
            else abspath(model_params['data_path']))

        # Outcome model - DATA FILTER
        feature_filter = model_params.get(
            'feature_filter', {'features': [], 'filter_mode': 'remove'})

        self.features_lwidget.addItems(feature_filter.get('features', []))
        for index in range(self.features_lwidget.count()):
            self.features_lwidget.item(index).setCheckState(2)

        self.filter_mode_cbox.setCurrentText(
            feature_filter.get('filter_mode', 'remove'))

        self.label_name_ledit.setText(model_params['label_name'])
        self.label_lower_bound_ledit.setText(
            '' if not model_params.get('label_bounds')
            or model_params['label_bounds'][0] == 1.0
            else str(model_params['label_bounds'][0]))
        self.label_upper_bound_ledit.setText(
            '' if not model_params.get('label_bounds')
            or model_params['label_bounds'][1] == 1.0
            else str(model_params['label_bounds'][1]))
        self.time_variable_ledit.setText(
            '' if not model_params.get('time_variable_name')
            else model_params['time_variable_name'])
        self.label_viewpoint_cbox.setCurrentText(
            model_params.get('label_viewpoint', 'longitudinal'))

        # Outcome model - FEATURE MAP
        self.fuzzy_matching_check.setCheckState(
            2 if model_params.get('fuzzy_matching', True) else 0)

        # Outcome model - MODEL FITTING, INSPECTION & EVALUATION
        self.prep_steps_ledit.setText(
            '[Identity]' if model_params.get('preprocessing_steps', []) == []
            else str(model_params['preprocessing_steps']).replace("\'", ''))

        tune_space = model_params.get(
            'tune_space', {
                'C': [2**-5, 2**10],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'degree': [3, 4, 5, 6],
                'gamma': [2**-15, 2**3],
                'tol': [1e-4, 1e-5, 1e-6],
                'class_weight': [None, 'balanced']})

        self.C_lower_bound_ledit.setText(
            '' if not tune_space.get('C') or tune_space['C'][0] == 2**-5
            else str(tune_space['C'][0]))
        self.C_upper_bound_ledit.setText(
            '' if not tune_space.get('C') or tune_space['C'][1] == 2**10
            else str(tune_space['C'][1]))

        for index in range(self.kernel_lwidget.count()):

            if (not tune_space.get('kernel')
                or self.kernel_lwidget.item(index).text()
                    in tune_space['kernel']):

                self.kernel_lwidget.item(index).setCheckState(2)

            else:

                self.kernel_lwidget.item(index).setCheckState(0)

        self.degree_ledit.setText(
            '' if not tune_space.get('degree')
            or tune_space['degree'] == [3, 4, 5, 6]
            else str(tune_space['degree']))

        self.gamma_lower_bound_ledit.setText(
            '' if not tune_space.get('gamma')
            or tune_space['gamma'][0] == 2**-15
            else str(tune_space['gamma'][0]))
        self.gamma_upper_bound_ledit.setText(
            '' if not tune_space.get('gamma')
            or tune_space['gamma'][1] == 2**3
            else str(tune_space['gamma'][1]))

        self.tol_ledit.setText(
            '' if not tune_space.get('tol')
            or tune_space['tol'] == [1e-4, 1e-5, 1e-6]
            else str(tune_space['tol']))

        for index in range(self.class_weight_lwidget.count()):

            if (not tune_space.get('class_weight')
                or self.class_weight_lwidget.item(index).text()
                    in map(str, tune_space['class_weight'])):

                self.class_weight_lwidget.item(index).setCheckState(2)

            else:

                self.class_weight_lwidget.item(index).setCheckState(0)

        self.tune_eval_sbox.setValue(model_params.get('tune_evaluations', 50))
        self.tune_score_cbox.setCurrentText(
            model_params.get('tune_score', 'Logloss'))
        self.tune_splits_sbox.setValue(model_params.get('tune_splits', 5))
        self.oof_splits_sbox.setValue(model_params.get('oof_splits', 5))
        self.write_features_check.setCheckState(
            2 if model_params.get('write_features', True) else 0)
        self.inspect_model_check.setCheckState(
            2 if model_params.get('inspect_model', True) else 0)
        self.evaluate_model_check.setCheckState(
            2 if model_params.get('evaluate_model', True) else 0)

        # Outcome model - VISUALIZATION
        display_options = model_params.get(
            'display_options', {
                'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                'kpis': ['Logloss', 'Brier score', 'Subset accuracy',
                         'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                         'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']})

        for index in range(self.graphs_lwidget.count()):

            if (not display_options.get('graphs')
                or self.graphs_lwidget.item(index).text()
                    in display_options['graphs']):

                self.graphs_lwidget.item(index).setCheckState(2)

            else:

                self.graphs_lwidget.item(index).setCheckState(0)

        for index in range(self.kpi_lwidget.count()):

            if (not display_options.get('kpis')
                or self.kpi_lwidget.item(index).text()
                    in display_options['kpis']):

                self.kpi_lwidget.item(index).setCheckState(2)

            else:

                self.kpi_lwidget.item(index).setCheckState(0)

        # Handling & Visualization
        self.identifier_ledit.setText('' if not identifier else identifier)
        self.disp_component_check.setCheckState(2 if display else 0)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'segment_ledit', 'link_ledit', 'weight_ledit', 'lower_bound_ledit',
            'upper_bound_ledit', 'model_label_ledit', 'data_path_ledit',
            'label_name_ledit', 'label_lower_bound_ledit',
            'label_upper_bound_ledit', 'time_variable_ledit',
            'prep_steps_ledit', 'identifier_ledit'))
        self.set_zero_line_cursor((
            'C_lower_bound_ledit', 'C_upper_bound_ledit', 'degree_ledit',
            'gamma_lower_bound_ledit', 'gamma_upper_bound_ledit',
            'tol_ledit'))

    def save(self):
        """."""

        # 
        model_parameters = {
            'model_label': self.model_label_ledit.text(),
            'model_folder_path': (
                None if self.model_path_ledit.text() == ''
                else abspath(self.model_path_ledit.text())),
            'data_path': (
                '' if self.data_path_ledit.text() == ''
                else abspath(self.data_path_ledit.text())),
            'feature_filter': {
                'features': [
                    self.features_lwidget.item(index).text()
                    for index in range(self.features_lwidget.count())
                    if self.features_lwidget.item(index).checkState()],
                'filter_mode': self.filter_mode_cbox.currentText()},
            'label_name': self.label_name_ledit.text(),
            'label_bounds': [
                1.0 if bound == '' else None if bound == 'None'
                else float(bound) for bound in (
                    self.label_lower_bound_ledit.text(),
                    self.label_upper_bound_ledit.text())],
            'time_variable_name': (
                None if self.time_variable_ledit.text() == ''
                else self.time_variable_ledit.text()),
            'label_viewpoint': self.label_viewpoint_cbox.currentText(),
            'fuzzy_matching': self.fuzzy_matching_check.isChecked(),
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
                'kernel': [
                    self.kernel_lwidget.item(index).text()
                    for index in range(self.kernel_lwidget.count())
                    if self.kernel_lwidget.item(index).checkState()],
                'degree': (
                    [3, 4, 5, 6] if self.degree_ledit.text() == ''
                    else loads(self.degree_ledit.text())),
                'gamma': (
                    [2**-15, 2**3] if any(bound == '' for bound in (
                        self.gamma_lower_bound_ledit.text(),
                        self.gamma_upper_bound_ledit.text()))
                    else [float(self.gamma_lower_bound_ledit.text()),
                          float(self.gamma_upper_bound_ledit.text())]),
                'tol': (
                    [1e-4, 1e-5, 1e-6] if self.tol_ledit.text() == ''
                    else loads(self.tol_ledit.text())),
                'class_weight': [
                    None
                    if self.class_weight_lwidget.item(index).text() == 'None'
                    else self.class_weight_lwidget.item(index).text()
                    for index in range(self.class_weight_lwidget.count())
                    if self.class_weight_lwidget.item(index).checkState()]},
            'tune_evaluations': self.tune_eval_sbox.value(),
            'tune_score': self.tune_score_cbox.currentText(),
            'tune_splits': self.tune_splits_sbox.value(),
            'inspect_model': self.inspect_model_check.isChecked(),
            'evaluate_model': self.evaluate_model_check.isChecked(),
            'oof_splits': self.oof_splits_sbox.value(),
            'write_features': self.write_features_check.isChecked(),
            'display_options': {
                'graphs': [
                    self.graphs_lwidget.item(index).text()
                    for index in range(self.graphs_lwidget.count())
                    if self.graphs_lwidget.item(index).checkState()],
                'kpis': [
                    self.kpi_lwidget.item(index).text()
                    for index in range(self.kpi_lwidget.count())
                    if self.kpi_lwidget.item(index).checkState()]}}

        # 
        component = {
            self.segment_ledit.text(): {
                'type': self.type_cbox.currentText(),
                'instance': {
                    'class': 'Support Vector Machine TCP',
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
                            None if self.link_ledit.text() == ''
                            else self.link_ledit.text().strip('][').split(', ')
                            ),
                        'identifier': (
                            None if self.identifier_ledit.text() == ''
                            else self.identifier_ledit.text()),
                        'display': self.disp_component_check.isChecked()}}}}

        # 
        value = component[self.segment_ledit.text()]

        # Check if the component is an objective
        if value['type'] == 'objective':

            # Set the icon path on the red target
            icon_path = (":/special_icons/icons_special/"
                         "target-red-svgrepo-com.svg")

        else:

            # Set the icon path on the red frame
            icon_path = (":/special_icons/icons_special/"
                         "frame-red-svgrepo-com.svg")

        # Initialize the icon object
        icon = QIcon()

        # Add the pixmap to the icon
        icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

        # Get the identifier string
        identifier = value['instance']['parameters']['identifier']

        # Get the embedding string
        embedding = ''.join(
            ('embedding: ', str(value['instance']['parameters']['embedding'])))

        # Get the weight string
        weight = ''.join(
            ('weight: ', str(value['instance']['parameters']['weight'])))

        # Join the segment and class name
        component_string = ' - '.join((substring for substring in (
            self.segment_ledit.text(), value['instance']['class'], identifier,
            embedding, weight) if substring))

        # 
        if self.parent.components_lwidget.currentItem():

            #
            del self.parent.plan_components[self.parent.plan_ledit.text()][
                self.parent.components_lwidget.currentItem().text()]

            # Remove the item from the list widget
            self.parent.components_lwidget.takeItem(
                self.parent.components_lwidget.currentRow())

            # Clear the selection in the list widget
            self.parent.components_lwidget.selectionModel().clear()

            # 
            self.parent.set_disabled(
                ('components_minus_tbutton', 'components_edit_tbutton'))

        # Add the icon with the component string to the list
        self.parent.components_lwidget.addItem(
            QListWidgetItem(icon, component_string))

        # 
        self.parent.plan_components[self.parent.plan_ledit.text()][
            component_string] = component

        # 
        self.close()

    def eventFilter(
            self,
            source,
            event):
        """
        Customize the event filters.

        Parameters
        ----------
        source : ...
            ...

        event : ...
            ...

        Returns
        -------
        ...
        """

        # Check if a mouse wheel event applies to QComboBox or QSpinBox
        if (event.type() == QEvent.Wheel and
                isinstance(source, (QComboBox, QSpinBox))):

            # Filter the event
            return True

        # Else, return the event
        return super().eventFilter(source, event)

    def mousePressEvent(self, event):

        super(QListWidget, self.features_lwidget).mousePressEvent(event)
        super(QListWidget, self.kernel_lwidget).mousePressEvent(event)
        super(QListWidget, self.class_weight_lwidget).mousePressEvent(event)
        super(QListWidget, self.graphs_lwidget).mousePressEvent(event)
        super(QListWidget, self.kpi_lwidget).mousePressEvent(event)

        if not self.features_lwidget.indexAt(event.pos()).isValid():
            self.features_lwidget.clearSelection()

        if not self.kernel_lwidget.indexAt(event.pos()).isValid():
            self.kernel_lwidget.clearSelection()

        if not self.class_weight_lwidget.indexAt(event.pos()).isValid():
            self.class_weight_lwidget.clearSelection()

        if not self.graphs_lwidget.indexAt(event.pos()).isValid():
            self.graphs_lwidget.clearSelection()

        if not self.kpi_lwidget.indexAt(event.pos()).isValid():
            self.kpi_lwidget.clearSelection()

    def add_model_path(self):
        """Add the model path from a snapshot folder."""

        # Get the loading folder path
        self.model_path_ledit.setText(QFileDialog.getExistingDirectory(
            self, 'Select a directory for loading'))

        # Set the model path field cursor position to zero
        self.model_path_ledit.setCursorPosition(0)

    def add_data_path(self):
        """Add the data path."""

        # Get the loading folder path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an outcome data file', QDir.rootPath(),
            'Outcome data (*.csv)')

        # 
        self.data_path_ledit.setText(abspath(path))

        # Set the model path field cursor position to zero
        self.data_path_ledit.setCursorPosition(0)

    def remove_feature(self):
        """Remove the selected feature from the model."""

        # Remove the item from the list widget
        self.features_lwidget.takeItem(self.features_lwidget.currentRow())

        # Clear the selection in the list widget
        self.features_lwidget.selectionModel().clear()

        # Disable specific fields
        self.set_disabled(('features_minus_tbutton',))

    def update_by_data_path(self):
        """."""

        # 
        data_path = self.data_path_ledit.text()

        # 
        if (data_path != '' and data_path.endswith('.csv')
                and isfile(data_path)):

            # 
            self.features_load_tbutton.setEnabled(True)

        else:

            # 
            self.features_load_tbutton.setEnabled(False)

    def update_save_button(self):
        """."""

        # 
        if any(text == '' for text in (
                self.segment_ledit.text(), self.model_label_ledit.text(),
                self.data_path_ledit.text(), self.label_name_ledit.text())):

            # 
            self.save_component_pbutton.setEnabled(False)

        else:

            # 
            self.save_component_pbutton.setEnabled(True)

    def open_input_dialog(self):
        """Open an input dialog."""

        self.setStyleSheet(
            "QInputDialog {background-color: rgb(238, 238, 236);}")

        # 
        text, accepted = QInputDialog.getText(
            self, 'pyanno4rt', 'Enter the feature name:')

        # 
        if (accepted and text and text not in [
                self.features_lwidget.item(index)
                for index in range(self.features_lwidget.count())]):

            # 
            item = QListWidgetItem(text)
            item.setCheckState(2)
            self.features_lwidget.addItem(item)

    def open_question_dialog(self):
        """Open a question dialog."""

        # Initialize the dictionary with the messages and function calls
        sources = {
            'features_load_tbutton': (
                "Loading the features from the data path will override the "
                "current feature set. Are you sure you want to proceed?",
                self.load_features_from_data)
            }

        # Get the message and function call by the attribute argument
        message, call = sources[self.sender().objectName()]

        # Check if the question dialog is confirmed
        if (QMessageBox.question(self, 'pyanno4rt', message)
                == QMessageBox.Yes):

            # Call the function
            call()

    def load_features_from_data(self):
        """."""

        # 
        feature_names = list(read_csv(self.data_path_ledit.text()).columns)

        # 
        self.features_lwidget.clear()

        # 
        self.features_lwidget.addItems(feature_names)
        for index in range(self.features_lwidget.count()):
            self.features_lwidget.item(index).setCheckState(2)

        # 
        self.filter_mode_cbox.setCurrentText('retain')

    def set_enabled(
            self,
            fieldnames):
        """
        Enable multiple fields by their names.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set it enabled
            getattr(self, name).setEnabled(True)

    def set_disabled(
            self,
            fieldnames):
        """
        Disable multiple fields by their names.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set it disabled
            getattr(self, name).setEnabled(False)

    def set_zero_line_cursor(
            self,
            fieldnames):
        """
        Set the line edit cursor positions to zero.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set the cursor position to zero
            getattr(self, name).setCursorPosition(0)

    def set_styles(
            self,
            key_value_pairs):
        """
        Set the element stylesheets from key-value pairs.

        Parameters
        ----------
        key_value_pairs : ...
            ...
        """

        # Loop over the dictionary items
        for key, value in key_value_pairs.items():

            # Get the attribute and set the stylesheet
            getattr(self, key).setStyleSheet(value)

    def close(self):
        """."""

        # 
        self.hide()
