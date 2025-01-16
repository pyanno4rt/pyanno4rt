"""Plan creation window."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from os.path import abspath, dirname, isfile
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QMainWindow, QMenu, QMessageBox, QSpinBox)

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.compilations.plan_creation_window import (
    Ui_plan_creation_window)
from pyanno4rt.gui.styles._custom_styles import (
    cbox, ledit, pbutton_composer, tbutton_composer)
from pyanno4rt.gui.windows.components import component_window_map
from pyanno4rt.optimization.components import component_map
from pyanno4rt.tools import load_segments_from_path

# %% Class definition


class PlanCreationWindow(QMainWindow, Ui_plan_creation_window):
    """
    Plan creation window for the GUI.

    This class sets up a plan creation window for the graphical user \
    interface, including input fields to define a plan.

    Parameters
    ----------
    parent : object of class \
        :class:`~pyanno4rt.gui.windows._main_window.MainWindow`
        The object representing the parent window for embedding.
    """

    def __init__(
            self,
            parent):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI plan creation window
        self.setupUi(self)

        # Get the parent window
        self.parent = parent

        # Initialize the segment dictionary
        self.segments = {}

        # Initialize the current component window
        self.current_component_window = None

        # Initialize the plan component dictionary
        self.plan_components = {}

        # Add the dropdown menu to the components 'plus' button
        self.add_dropdown_to_components()

        # Disable specific fields
        self.set_disabled((
            'components_plus_tbutton', 'components_minus_tbutton',
            'components_edit_tbutton'))

        # Set the stylesheets
        self.set_styles({
            'plan_ledit': ledit,
            'ref_plan_cbox': cbox,
            'modality_cbox': cbox,
            'img_path_ledit': ledit,
            'img_path_tbutton': tbutton_composer,
            'dose_path_ledit': ledit,
            'dose_path_tbutton': tbutton_composer,
            'dose_res_ledit_x': ledit,
            'dose_res_ledit_y': ledit,
            'dose_res_ledit_z': ledit,
            'components_plus_tbutton': tbutton_composer,
            'components_minus_tbutton': tbutton_composer,
            'components_edit_tbutton': tbutton_composer,
            'create_plan_pbutton': pbutton_composer,
            'close_plan_pbutton': pbutton_composer})

        # Install the custom event filter for the reference combo box
        self.ref_plan_cbox.installEventFilter(self)

        # Adjust the component list widget spacing
        self.components_lwidget.setSpacing(4)

        # Overwrite the whee event of the component list widget
        self.components_lwidget.wheelEvent = lambda event: None

        # Connect the fields with the event signals
        self.connect_signals()

    def eventFilter(
            self,
            source,
            event):
        """
        Filter the events (overwrites the default event filter).

        Parameters
        ----------
        source : object of class :class:`~PyQt5.QtWidgets`
            The object representing the event source.

        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.

        Returns
        -------
        bool or object of class :class:`~PyQt5.QtCore.QEvent`
            Boolean value or event object depending on the filter.
        """

        # Check if a mouse wheel event applies to QComboBox or QSpinBox
        if (event.type() == QEvent.Wheel and
                isinstance(source, (QComboBox, QSpinBox))):

            # Filter the event by returning True
            return True

        # Else, return the unfiltered event
        return super().eventFilter(source, event)

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

        # Check if no element of the components list widget has been clicked
        if not self.components_lwidget.indexAt(event.pos()).isValid():

            # Clear the element selection
            self.components_lwidget.clearSelection()

            # Disable the 'minus' and 'edit' buttons
            self.set_disabled((
                'components_minus_tbutton', 'components_edit_tbutton'))

    def add_dropdown_to_components(self):
        """Add the dropdown menu to the components 'plus' button."""

        # Initialize the dropdown menu
        menu = QMenu()

        # Loop over the component map keys
        for key in component_map:

            # Add the key to the dropdown menu
            menu.addAction(key, partial(self.open_component_window, key))

        # Connect the menu trigger event
        menu.triggered.connect(self.update_plan_component_key)

        # Set the popup mode for the component 'plus' button
        self.components_plus_tbutton.setPopupMode(2)

        # Add the dropdown menu to the 'plus' button
        self.components_plus_tbutton.setMenu(menu)

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
            'img_path_tbutton': self.add_imaging_path,
            'dose_path_tbutton': self.add_dose_matrix_path,
            'components_minus_tbutton': self.remove_component,
            'components_edit_tbutton': self.edit_component,
            'create_plan_pbutton': self.create,
            'close_plan_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Loop over the field names with 'textChanged' events
        for key in (
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'dose_res_ledit_x', 'dose_res_ledit_y', 'dose_res_ledit_z'
                ):

            # Connect the 'textChanged' event
            getattr(self, key).textChanged.connect(self.update_fields)

        # Loop over the field names with 'itemClicked' events
        for key, value in {
            'components_lwidget': (lambda: self.set_enabled((
                'components_minus_tbutton', 'components_edit_tbutton')))
                }.items():

            # Connect the 'itemClicked' event
            getattr(self, key).itemClicked.connect(value)

        # Connect the 'currentTextChanged' event with the reference combo box
        self.ref_plan_cbox.currentTextChanged.connect(self.update_fields)

        # Connect the 'rowsInserted'/'rowsRemoved' events with the list widget
        self.components_lwidget.model().rowsInserted.connect(
            self.update_fields)
        self.components_lwidget.model().rowsRemoved.connect(
            self.update_fields)

    def add_imaging_path(self):
        """Add the CT and segmentation data from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a patient data file', '',
            'CT/Segmentation data (*.dcm *.mat *.p)')

        # Check if the file path exists
        if path:

            # Check if a DICOM file is selected
            if path.endswith('.dcm'):

                # Get the directory path
                path = dirname(path)

            # Set the imaging path field
            self.img_path_ledit.setText(abspath(path))

            # Set the imaging path field cursor position to zero
            self.img_path_ledit.setCursorPosition(0)

    def add_dose_matrix_path(self):
        """Add the dose-influence matrix from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a dose-influence matrix file', '',
            'Dose-influence matrix (*.mat *.npy)')

        # Check if the file path exists
        if path:

            # Set the dose path field
            self.dose_path_ledit.setText(abspath(path))

            # Set the dose path field cursor position to zero
            self.dose_path_ledit.setCursorPosition(0)

    def open_component_window(
            self,
            name):
        """
        Open a component window.

        Parameters
        ----------
        name : str
            Name of the component window.
        """

        # Get the component window
        self.current_component_window = component_window_map[name](self)

        # Set the position of the window
        self.current_component_window.position()

        # Show the window
        self.current_component_window.show()

    def remove_component(self):
        """Remove the selected component."""

        # Remove the component from the GUI components dictionary
        del self.plan_components[next(iter(self.plan_components))][
            self.components_lwidget.currentItem().text()]

        # Remove the item from the list widget
        self.components_lwidget.takeItem(self.components_lwidget.currentRow())

        # Clear the selection in the list widget
        self.components_lwidget.selectionModel().clear()

        # Disable specific fields
        self.set_disabled((
            'components_minus_tbutton', 'components_edit_tbutton'))

    def edit_component(self):
        """Edit the selected component."""

        # Get the selected component
        component = self.plan_components[next(iter(self.plan_components))][
            self.components_lwidget.currentItem().text()]

        # Loop over the component values
        for value in component.values():

            # Get the component window
            self.current_component_window = component_window_map[
                value['instance']['function']](self)

        # Set the position of the window
        self.current_component_window.position()

        # Load the component parameters into the window
        self.current_component_window.load(component)

        # Show the window
        self.current_component_window.show()

    def create(self):
        """Create the new treatment plan."""

        # Get the treatment plan label
        new_label = self.plan_ledit.text()

        # Get the reference plan
        reference = self.ref_plan_cbox.currentText()

        # Check if a reference plan has been selected
        if reference != 'None':

            # Copy the reference input dictionaries
            configuration = self.parent.plans[reference].configuration.copy()
            optimization = self.parent.plans[reference].optimization.copy()
            evaluation = self.parent.plans[reference].evaluation.copy()

            # Change the treatment plan label
            configuration['label'] = new_label

            # Initialize the treatment plan
            new_plan = TreatmentPlan(configuration, optimization, evaluation)

            # Activate the new treatment plan in the main window
            self.parent.activate(new_plan)

        else:

            # Reset the parent selector index to the default
            self.parent.plan_select_cbox.setCurrentIndex(-1)

            # Transfer the treatment plan label
            self.parent.plan_ledit.setText(new_label)

            # Transfer the treatment modality
            self.parent.modality_cbox.setCurrentText(
                self.modality_cbox.currentText())

            # Transfer the imaging path
            self.parent.img_path_ledit.setText(
                abspath(self.img_path_ledit.text()))

            # Transfer the dose path
            self.parent.dose_path_ledit.setText(
                abspath(self.dose_path_ledit.text()))

            # Transfer the dose resolution
            self.parent.dose_res_ledit_x.setText(self.dose_res_ledit_x.text())
            self.parent.dose_res_ledit_y.setText(self.dose_res_ledit_y.text())
            self.parent.dose_res_ledit_z.setText(self.dose_res_ledit_z.text())

            # Transfer the plan components
            self.parent.plan_components |= self.plan_components

            # Loop over the component list widget items
            for i in range(self.components_lwidget.count()):

                # Get a clone of the item
                item_clone = self.components_lwidget.item(i).clone()

                # Transfer the item clone
                self.parent.components_lwidget.addItem(item_clone)

            # Initialize the treatment plan from the main window
            self.parent.initialize()

            # Load the segment names and types from the imaging path
            self.segments = load_segments_from_path(self.img_path_ledit.text())

        # Close the plan creation window
        self.close()

    def update_fields(self):
        """Update the plan creator fields by condition."""

        # Check if a reference plan has been selected
        if self.ref_plan_cbox.currentText() != 'None':

            # Disable specific fields
            self.set_disabled((
                'modality_cbox', 'img_path_ledit', 'img_path_tbutton',
                'dose_path_ledit', 'dose_path_tbutton', 'dose_res_ledit_x',
                'dose_res_ledit_y', 'dose_res_ledit_z', 'components_lwidget'))

        else:

            # Enable specific fields
            self.set_enabled((
                'modality_cbox', 'img_path_ledit', 'img_path_tbutton',
                'dose_path_ledit', 'dose_path_tbutton', 'dose_res_ledit_x',
                'dose_res_ledit_y', 'dose_res_ledit_z', 'components_lwidget'))

        # Check if the imaging path leads to a file
        if isfile(self.img_path_ledit.text()):

            try:

                # Load the segment names and types
                self.segments = load_segments_from_path(
                    self.img_path_ledit.text())

                # Set the boolean indicator to True
                loaded_segments = True

            except Exception as exception:

                # Reset the imaging path field
                self.img_path_ledit.setText('')

                # Show a warning message box
                QMessageBox.warning(self, "pyanno4rt", str(exception))

                # Set the boolean indicator to False
                loaded_segments = False

        else:

            # Set the boolean indicator to False by default
            loaded_segments = False

        # Check if any condition blocks the addition of components
        if (self.plan_ledit.text() == ''
                or self.ref_plan_cbox.currentText() != 'None'
                or not loaded_segments):

            # Disable the component 'plus' button
            self.components_plus_tbutton.setEnabled(False)

        else:

            # Enable the component 'plus' button
            self.components_plus_tbutton.setEnabled(True)

        # Check if any condition blocks the treatment plan creation
        if (((any(text == '' for text in (
                self.plan_ledit.text(), self.img_path_ledit.text(),
                self.dose_path_ledit.text(), self.dose_res_ledit_x.text(),
                self.dose_res_ledit_y.text(), self.dose_res_ledit_z.text()))
                or self.components_lwidget.count() == 0)
                and self.ref_plan_cbox.currentText() == 'None')
            or
            ((self.plan_ledit.text() == ''
              or self.plan_ledit.text() in (
                  self.parent.plan_select_cbox.itemText(i)
                  for i in range(self.parent.plan_select_cbox.count())))
             and self.ref_plan_cbox.currentText() != 'None')):

            # Disable the plan creation button
            self.create_plan_pbutton.setEnabled(False)

        else:

            # Enable the plan creation button
            self.create_plan_pbutton.setEnabled(True)

    def update_plan_component_key(self):
        """Update the plan component dictionary key."""

        # Check if the dictionary has a single key
        if len(self.plan_components) == 1:

            # Check if the key is different from the current plan label
            if next(iter(self.plan_components)) != self.plan_ledit.text():

                # Move the components to the current plan label
                self.plan_components[self.plan_ledit.text()] = (
                    self.plan_components[next(iter(self.plan_components))])

                # Delete the previous key
                del self.plan_components[next(iter(self.plan_components))]

        else:

            # Initialize the component subdictionary for the plan label
            self.plan_components[self.plan_ledit.text()] = {}

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the plan creation window."""

        # Reset the plan component dictionary
        self.plan_components = {}

        # Hide the window
        self.hide()
