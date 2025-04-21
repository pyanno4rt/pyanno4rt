"""Equivalent uniform dose component window."""

# Author: Tim Ortkamp

# %% External package import

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QMainWindow

# %% Internal package import

from pyanno4rt.gui._custom_styles import cbox, ledit, pbutton_composer, sbox
from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.compilations.components.equivalent_uniform_dose_window import (
    Ui_equivalent_uniform_dose_window)
from pyanno4rt.gui.custom_widgets import CheckableComboBox
from pyanno4rt.optimization.components import EquivalentUniformDose
from pyanno4rt.tools import string_to_numeric

# %% Class definition


class EquivalentUniformDoseWindow(
        QMainWindow, Ui_equivalent_uniform_dose_window):
    """
    Equivalent uniform dose component window for the GUI.

    This class sets up the equivalent uniform dose component window for the \
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

        # Initialize the edit boolean
        self.edit = False

        # Initialize the checkable segment link combo box
        self.segment_link_cbox = CheckableComboBox()

        # Set a fixed size
        self.segment_link_cbox.setFixedSize(391, 31)

        # Add the items
        self.segment_link_cbox.addItems(list(self.parent.segments), False)

        # Add the combo box to the layout
        self.segment_link_layout.addWidget(self.segment_link_cbox)

        # Add the segment items to the segment combo box
        self.segment_cbox.addItems(list(self.parent.segments))
        self.segment_cbox.setCurrentIndex(-1)

        # Set the stylesheets
        self.set_styles({
            'segment_cbox': cbox,
            'type_cbox': cbox,
            'embedding_cbox': cbox,
            'target_eud_ledit': ledit,
            'vol_eff_ledit': ledit,
            'segment_link_cbox': cbox,
            'weight_ledit': ledit,
            'rank_sbox': sbox,
            'lower_bound_ledit': ledit,
            'upper_bound_ledit': ledit,
            'identifier_ledit': ledit,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox and QSpinBox elements
        for box in (
                'segment_cbox', 'type_cbox', 'embedding_cbox',
                'segment_link_cbox', 'rank_sbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(parent)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'target_eud_ledit', 'vol_eff_ledit', 'weight_ledit',
            'lower_bound_ledit', 'upper_bound_ledit', 'identifier_ledit'))

        # Disable the save button
        self.save_pbutton.setEnabled(False)

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
                'save_pbutton': self.save,
                'close_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Loop over the field names with 'currentTextChanged' events
        for key, value in {
                'segment_cbox': self.update_save_button
                }.items():

            # Connect the 'currentTextChanged' event
            getattr(self, key).currentTextChanged.connect(value)

        # Loop over the field names with 'textChanged' events
        for key, value in {
                'target_eud_ledit': self.update_save_button,
                'vol_eff_ledit': self.update_save_button
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
            :class:`~pyanno4rt.optimization.components._equivalent_uniform_dose.EquivalentUniformDose`
            The object used to represent the optimization component.

        edit : bool
            Indicator for the editing of the component.
        """

        # Get the edit attribute from the argument
        self.edit = edit

        # Get the component parameters
        (segment, target_eud, volume_parameter, component_type, embedding,
         weight, rank, bounds, link, identifier, display) = (
             component.arguments.values())

        # Convert the bounds
        lower, upper = (None, None) if bounds is None else bounds

        # Loop over the fields with 'setText' method
        for key, value in {
                'target_eud_ledit': str(target_eud),
                'vol_eff_ledit': str(volume_parameter),
                'weight_ledit': '' if weight == 1.0 else str(weight),
                'lower_bound_ledit': '' if lower is None else str(lower),
                'upper_bound_ledit': '' if upper is None else str(upper),
                'identifier_ledit': '' if not identifier else identifier
                }.items():

            # Set the text
            getattr(self, key).setText(value)

        # Loop over the fields with 'setCurrentText' method
        for key, value in {
                'segment_cbox': segment,
                'type_cbox': component_type,
                'embedding_cbox': embedding,
                }.items():

            # Set the text
            getattr(self, key).setCurrentText(value)

        # Loop over the fields with 'setValue' method
        for key, value in {
                'rank_sbox': rank,
                }.items():

            # Set the value
            getattr(self, key).setValue(value)

        # Loop over the fields with 'setCheckState' method
        for key, value in {
                'disp_component_check': 2*display
                }.items():

            # Set the check state
            getattr(self, key).setCheckState(value)

        # Loop over the checkable combo boxes with their selections
        for box, selection in {
                'segment_link_cbox': [] if not link else link
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
            'target_eud_ledit', 'vol_eff_ledit', 'weight_ledit',
            'lower_bound_ledit', 'upper_bound_ledit', 'identifier_ledit'))

    def save(self):
        """Save the fields to a component."""

        # Get the component
        component = EquivalentUniformDose(
            segment=self.segment_cbox.currentText(),
            target_eud=string_to_numeric(self.target_eud_ledit.text()),
            volume_parameter=string_to_numeric(self.vol_eff_ledit.text()),
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

    def update_save_button(self):
        """Update the save button."""

        # Enable or disable the save button
        self.save_pbutton.setEnabled(
            all(text != '' for text in (
                    self.segment_cbox.currentText(),
                    self.target_eud_ledit.text(),
                    self.vol_eff_ledit.text())))

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the equivalent uniform dose window."""

        # Reset the edit boolean
        self.edit = False

        # Hide the window
        self.hide()
