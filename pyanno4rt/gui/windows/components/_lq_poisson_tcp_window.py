"""LQ Poisson TCP component window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QComboBox, QListWidgetItem, QMainWindow, QSpinBox

# %% Internal package import

from pyanno4rt.gui.compilations.components.lq_poisson_tcp_window import (
    Ui_lq_poisson_tcp_window)
from pyanno4rt.gui.styles._custom_styles import ledit, pbutton_composer

# %% Class definition


class LQPoissonTCPWindow(QMainWindow, Ui_lq_poisson_tcp_window):
    """
    LQ Poisson TCP component window for the application.

    This class creates a LQ Poisson TCP component window for the graphical \
    user interface, including input fields to parametrize the component.
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
        for box in ('type_cbox', 'embedding_cbox', 'rank_sbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(self)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'segment_ledit', 'link_ledit', 'identifier_ledit'))

        # 
        self.save_component_pbutton.setEnabled(False)

        # 
        self.set_styles({'segment_ledit': ledit,
                         'link_ledit': ledit,
                         'identifier_ledit': ledit,
                         'save_component_pbutton': pbutton_composer,
                         'close_component_pbutton': pbutton_composer})

        # 
        self.segment_ledit.textChanged.connect(self.update_save_button)
        self.alpha_ledit.textChanged.connect(self.update_save_button)
        self.beta_ledit.textChanged.connect(self.update_save_button)
        self.vol_eff_ledit.textChanged.connect(self.update_save_button)

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
        alpha = component[key]['instance']['parameters']['alpha']
        beta = component[key]['instance']['parameters']['beta']
        vol_eff = component[key]['instance']['parameters']['volume_parameter']
        embedding = component[key]['instance']['parameters'].get(
            'embedding', 'active')
        weight = component[key]['instance']['parameters'].get('weight', 1)
        rank = component[key]['instance']['parameters'].get('rank', 1)
        lower, upper = component[key]['instance']['parameters'].get(
            'bounds', (None, None))
        link = component[key]['instance']['parameters'].get('link')
        identifier = component[key]['instance']['parameters'].get('identifier')
        display = component[key]['instance']['parameters'].get('display', True)

        # 
        self.segment_ledit.setText(key)

        # 
        self.type_cbox.setCurrentText(ctype)

        # 
        self.alpha_ledit.setText(str(float(alpha)))

        # 
        self.beta_ledit.setText(str(float(beta)))

        # 
        self.vol_eff_ledit.setText(str(float(vol_eff)))

        # 
        self.embedding_cbox.setCurrentText(embedding)

        # 
        self.weight_ledit.setText('' if weight == 1 else str(float(weight)))

        # 
        self.rank_sbox.setValue(rank)

        # 
        self.lower_bound_ledit.setText('' if not lower else str(float(lower)))

        # 
        self.upper_bound_ledit.setText('' if not upper else str(float(upper)))

        # 
        self.link_ledit.setText(
            '' if not link else str(link).replace("\'", ''))

        # 
        self.identifier_ledit.setText('' if not identifier else identifier)

        # 
        self.disp_component_check.setCheckState(2 if display else 0)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'segment_ledit', 'link_ledit', 'identifier_ledit'))

    def save(self):
        """."""

        # 
        component = {
            self.segment_ledit.text(): {
                'type': self.type_cbox.currentText(),
                'instance': {
                    'class': 'LQ Poisson TCP',
                    'parameters': {
                        'alpha': float(self.alpha_ledit.text()),
                        'beta': float(self.beta_ledit.text()),
                        'volume_parameter': float(self.vol_eff_ledit.text()),
                        'embedding': self.embedding_cbox.currentText(),
                        'weight': (
                            1.0 if self.weight_ledit.text() == ''
                            else float(self.weight_ledit.text())),
                        'rank': self.rank_sbox.value(),
                        'bounds': [
                            0 if self.lower_bound_ledit.text() == ''
                            else float(self.lower_bound_ledit),
                            1 if self.upper_bound_ledit.text() == ''
                            else float(self.upper_bound_ledit)],
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

        # Else, return the even
        return super().eventFilter(source, event)

    def update_save_button(self):
        """."""

        # 
        if any(text == '' for text in (
                self.segment_ledit.text(), self.alpha_ledit.text(),
                self.beta_ledit.text(), self.vol_eff_ledit.text())):

            # 
            self.save_component_pbutton.setEnabled(False)

        else:

            # 
            self.save_component_pbutton.setEnabled(True)

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
