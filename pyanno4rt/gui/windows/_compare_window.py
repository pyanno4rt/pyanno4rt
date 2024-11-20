"""Plan comparison window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.compare_window import Ui_compare_window
from pyanno4rt.gui.custom_widgets import DVHCompareWidget, SliceCompareWidget

# %% Class definition


class CompareWindow(QMainWindow, Ui_compare_window):
    """
    Plan comparison window for the GUI.

    This class creates the plan comparison window for the graphical user \
    interface, including some general information on the package.
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

        # 
        self.baseline_dose_slice_widget = SliceCompareWidget(self)
        self.reference_dose_slice_widget = SliceCompareWidget(self)
        self.difference_dose_slice_widget = SliceCompareWidget(self, 'seismic')

        # 
        self.baseline_dvh_widget = DVHCompareWidget(self)
        self.reference_dvh_widget = DVHCompareWidget(self)
        self.difference_dvh_widget = DVHCompareWidget(self)

        # 
        self.baseline_dose_layout.insertWidget(
            0, self.baseline_dose_slice_widget)
        self.reference_dose_layout.insertWidget(
            0, self.reference_dose_slice_widget)
        self.difference_dose_layout.insertWidget(
            0, self.difference_dose_slice_widget)

        # 
        self.baseline_dvh_layout.insertWidget(0, self.baseline_dvh_widget)
        self.reference_dvh_layout.insertWidget(0, self.reference_dvh_widget)
        self.difference_dvh_layout.insertWidget(0, self.difference_dvh_widget)

        # 
        self.slice_selection_sbar.valueChanged.connect(
            self.baseline_dose_slice_widget.change_image_slice)
        self.slice_selection_sbar.valueChanged.connect(
            self.reference_dose_slice_widget.change_image_slice)
        self.slice_selection_sbar.valueChanged.connect(
            self.difference_dose_slice_widget.change_image_slice)

        # 
        self.opacity_sbox.valueChanged.connect(
            self.baseline_dose_slice_widget.change_dose_opacity)
        self.opacity_sbox.valueChanged.connect(
            self.reference_dose_slice_widget.change_dose_opacity)
        self.opacity_sbox.valueChanged.connect(
            self.difference_dose_slice_widget.change_dose_opacity)

        # 
        self.close_compare_pbutton.clicked.connect(self.close)

    def add_plots(self, baseline, reference):
        """."""

        # 
        self.baseline_dose_slice_widget.reset_images()
        self.reference_dose_slice_widget.reset_images()
        self.difference_dose_slice_widget.reset_images()

        # 
        self.baseline_dose_slice_widget.add_ct(
            baseline, baseline.datahub.computed_tomography['cube'])
        self.reference_dose_slice_widget.add_ct(
            reference, reference.datahub.computed_tomography['cube'])
        self.difference_dose_slice_widget.add_ct(
            baseline, baseline.datahub.computed_tomography['cube'])

        # Get the axial dimension of the CT cube
        axial_length = baseline.datahub.computed_tomography[
            'cube_dimensions'][2]

        # Add the segments to the slice widget
        self.baseline_dose_slice_widget.add_segments(
            baseline.datahub.computed_tomography,
            baseline.datahub.segmentation)
        self.reference_dose_slice_widget.add_segments(
            reference.datahub.computed_tomography,
            reference.datahub.segmentation)
        self.difference_dose_slice_widget.add_segments(
            baseline.datahub.computed_tomography,
            baseline.datahub.segmentation)

        # Set the range of the slice selection scrollbar
        self.slice_selection_sbar.setRange(0, axial_length-1)

        # Set the initial scrollbar value
        self.slice_selection_sbar.setValue(int((axial_length-1)/2))

        # Set the initial position label
        self.slice_selection_pos.setText(''.join((
            'z = ', str(self.baseline_dose_slice_widget.slice), ' mm')))

        # Get the joint minimum and maximum dose
        minima = [min(
            baseline.datahub.optimization[
                'optimized_dose'][:, :, index].min(),
            reference.datahub.optimization[
                'optimized_dose'][:, :, index].min())
            for index in range(axial_length)]
        maxima = [max(
            baseline.datahub.optimization[
                'optimized_dose'][:, :, index].max(),
            reference.datahub.optimization[
                'optimized_dose'][:, :, index].max())
            for index in range(axial_length)]

        # Add the dose image to the slice widget
        self.baseline_dose_slice_widget.add_dose(
            baseline.datahub.optimization['optimized_dose'],
            minima, maxima)
        self.reference_dose_slice_widget.add_dose(
            reference.datahub.optimization['optimized_dose'],
            minima, maxima)

        # Get the dose difference
        dose_diff = (reference.datahub.optimization['optimized_dose']
                     - baseline.datahub.optimization['optimized_dose'])

        # Get the maximum absolute difference
        max_diff = max(abs(dose_diff.min()), abs(dose_diff.max()))

        # Add the dose difference image to the slice widget
        self.difference_dose_slice_widget.add_dose(
            dose_diff, [-max_diff], [max_diff])

        # 
        self.baseline_dose_slice_widget.update_images()
        self.reference_dose_slice_widget.update_images()
        self.difference_dose_slice_widget.update_images()

        # 
        self.baseline_dvh_widget.reset_dvh()
        self.reference_dvh_widget.reset_dvh()
        self.difference_dvh_widget.reset_dvh()

        # 
        dvh_diff = {
            segment: {'dvh_values': (
                reference.datahub.dose_histogram[segment]['dvh_values']
                - baseline.datahub.dose_histogram[segment]['dvh_values'])}
            for segment in baseline.datahub.dose_histogram
            if segment not in ('evaluation_points', 'display_segments')}
        dvh_diff |= {
            'evaluation_points': (
                baseline.datahub.dose_histogram['evaluation_points']),
            'display_segments': (
                baseline.datahub.dose_histogram['display_segments'])}

        # 
        joint_segments = tuple(
            set(baseline.datahub.dose_histogram['display_segments'])
            & set(reference.datahub.dose_histogram['display_segments']))

        # 
        if len(joint_segments) == 0:

            joint_segments = None

        # 
        diff_ranges = {
            'x': (0,
                  max(baseline.datahub.dose_histogram['evaluation_points'][-1],
                      reference.datahub.dose_histogram['evaluation_points'][-1]
                      )),
            'y': (
                min(min(
                    dvh_diff[segment]['dvh_values']) for segment in joint_segments
                    if segment not in ('evaluation_points', 'display_segments')),
                max(max(
                    dvh_diff[segment]['dvh_values']) for segment in joint_segments
                    if segment not in ('evaluation_points', 'display_segments')))}

        # Add the style and input data to the DVH widget
        self.baseline_dvh_widget.add_style_and_data(
            baseline.datahub.dose_histogram, x_range=diff_ranges['x'],
            baseline=baseline)
        self.reference_dvh_widget.add_style_and_data(
            reference.datahub.dose_histogram, x_range=diff_ranges['x'],
            reference=reference)
        self.difference_dvh_widget.add_style_and_data(
            dvh_diff, x_range=diff_ranges['x'], y_range=diff_ranges['y'],
            baseline=baseline, reference=reference)

        # Update the plot of the DVH widget
        self.baseline_dvh_widget.update_dvh(joint_segments)
        self.reference_dvh_widget.update_dvh(joint_segments)
        self.difference_dvh_widget.update_dvh(joint_segments)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def close(self):
        """."""

        # 
        self.hide()
