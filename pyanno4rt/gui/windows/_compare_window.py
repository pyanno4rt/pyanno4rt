"""Plan comparison window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import ceil, linspace
from PyQt5.QtWidgets import QMainWindow
from pyqtgraph import mkPen

# %% Internal package import

from pyanno4rt.gui.compilations.compare_window import Ui_compare_window
from pyanno4rt.gui.custom_widgets import DVHCompareWidget, SliceCompareWidget
from pyanno4rt.tools import get_constraint_segments, get_objective_segments

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

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Get the application from the argument
        self.parent = parent

        # 
        self.baseline, self.reference = None, None

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
        self.baseline_dose_slice_widget.viewbox.setXLink(
            self.reference_dose_slice_widget.viewbox)
        self.baseline_dose_slice_widget.viewbox.setYLink(
            self.reference_dose_slice_widget.viewbox)
        self.difference_dose_slice_widget.viewbox.setXLink(
            self.baseline_dose_slice_widget.viewbox)
        self.difference_dose_slice_widget.viewbox.setYLink(
            self.reference_dose_slice_widget.viewbox)

        # 
        self.baseline_dvh_widget.plot_graph.getPlotItem().vb.setXLink(
            self.reference_dvh_widget.plot_graph.getPlotItem().vb)
        self.baseline_dvh_widget.plot_graph.getPlotItem().vb.setYLink(
            self.reference_dvh_widget.plot_graph.getPlotItem().vb)

        # 
        self.orient_cbox.currentTextChanged.connect(
            self.adjust_slider_by_orientation)
        self.orient_cbox.currentTextChanged.connect(
            self.baseline_dose_slice_widget.change_orientation)
        self.orient_cbox.currentTextChanged.connect(
            self.reference_dose_slice_widget.change_orientation)
        self.orient_cbox.currentTextChanged.connect(
            self.difference_dose_slice_widget.change_orientation)

        # 
        self.opacity_sbox.valueChanged.connect(
            self.baseline_dose_slice_widget.change_dose_opacity)
        self.opacity_sbox.valueChanged.connect(
            self.reference_dose_slice_widget.change_dose_opacity)
        self.opacity_sbox.valueChanged.connect(
            self.difference_dose_slice_widget.change_dose_opacity)

        # 
        self.slice_selection_sbar.valueChanged.connect(
            self.baseline_dose_slice_widget.change_image_slice)
        self.slice_selection_sbar.valueChanged.connect(
            self.reference_dose_slice_widget.change_image_slice)
        self.slice_selection_sbar.valueChanged.connect(
            self.difference_dose_slice_widget.change_image_slice)

        # 
        self.disCT_cbox.stateChanged.connect(
            self.baseline_dose_slice_widget.toggle_ct)
        self.disCT_cbox.stateChanged.connect(
            self.reference_dose_slice_widget.toggle_ct)
        self.disCT_cbox.stateChanged.connect(
            self.difference_dose_slice_widget.toggle_ct)

        # 
        self.disDose_cbox.stateChanged.connect(
            self.baseline_dose_slice_widget.toggle_dose)
        self.disDose_cbox.stateChanged.connect(
            self.reference_dose_slice_widget.toggle_dose)
        self.disDose_cbox.stateChanged.connect(
            self.difference_dose_slice_widget.toggle_dose)

        # 
        self.disDoseCon_cbox.stateChanged.connect(
            self.baseline_dose_slice_widget.toggle_dose_contours)
        self.disDoseCon_cbox.stateChanged.connect(
            self.reference_dose_slice_widget.toggle_dose_contours)
        self.disDoseCon_cbox.stateChanged.connect(
            self.difference_dose_slice_widget.toggle_dose_contours)

        # 
        self.disSegm_cbox.stateChanged.connect(
            self.baseline_dose_slice_widget.toggle_segment_contours)
        self.disSegm_cbox.stateChanged.connect(
            self.reference_dose_slice_widget.toggle_segment_contours)
        self.disSegm_cbox.stateChanged.connect(
            self.difference_dose_slice_widget.toggle_segment_contours)

        # 
        self.joint_dvh_pbutton.clicked.connect(self.open_joint_dvh)

        # 
        self.close_compare_pbutton.clicked.connect(self.close)

    def add_plots(self, baseline, reference):
        """."""

        # 
        self.baseline, self.reference = baseline, reference

        # 
        self.baseline_dose_slice_widget.reset_images()
        self.reference_dose_slice_widget.reset_images()
        self.difference_dose_slice_widget.reset_images()

        # Get the joint minimum and maximum dose
        minimum = min(
            baseline.datahub.optimization['optimized_dose'].min(),
            reference.datahub.optimization['optimized_dose'].min())
        maximum = max(
            baseline.datahub.optimization['optimized_dose'].max(),
            reference.datahub.optimization['optimized_dose'].max())

        # 
        self.baseline_dose_slice_widget.add_image_data(
            baseline, minimum, maximum)
        self.reference_dose_slice_widget.add_image_data(
            reference, minimum, maximum)
        self.difference_dose_slice_widget.add_image_data(
            (baseline, reference), minimum, maximum)

        # 
        self.adjust_slider_by_orientation()

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
        joint_segments = sorted(tuple(
            set(baseline.datahub.dose_histogram['display_segments'])
            & set(reference.datahub.dose_histogram['display_segments'])),
            key=lambda t: t[0])

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
                    dvh_diff[segment]['dvh_values'])
                    for segment in joint_segments
                    if segment not in ('evaluation_points', 'display_segments')),
                max(max(
                    dvh_diff[segment]['dvh_values'])
                    for segment in joint_segments
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

    def adjust_slider_by_orientation(self):
        """."""

        # 
        if self.orient_cbox.currentText() == 'axial':

            # Get the axial dimension of the CT cube
            axial_length = self.baseline.datahub.computed_tomography[
                'cube_dimensions'][2]

        # 
        elif self.orient_cbox.currentText() == 'coronal':

            # Get the axial dimension of the CT cube
            axial_length = self.baseline.datahub.computed_tomography[
                'cube_dimensions'][0]

        else:

            # Get the axial dimension of the CT cube
            axial_length = self.baseline.datahub.computed_tomography[
                'cube_dimensions'][1]

        # Set the range of the slice selection scrollbar
        self.slice_selection_sbar.setRange(0, axial_length-1)

        # Set the initial scrollbar value
        self.slice_selection_sbar.setValue(int((axial_length-1)/2))

    def select_dvh_curves(self, event):
        """."""

        # 
        event_pen_width = event.curve.opts['pen'].width()

        for widget in (self.baseline_dvh_widget, self.reference_dvh_widget,
                       self.difference_dvh_widget):

            # Get all plot items
            items = widget.plot_graph.getPlotItem().listDataItems()

            for item in items:
                pen = item.curve.opts['pen']
                item.curve.setPen(mkPen(
                    color=pen.color(), style=pen.style(), width=1))
                if item != event and item.name() == event.name():
                    if event_pen_width != 4:
                        item.curve.setPen(mkPen(
                            color=pen.color(), style=pen.style(), width=2))
                    else:
                        item.curve.setPen(mkPen(
                            color=pen.color(), style=pen.style(), width=1))
                elif item == event:
                    if event_pen_width != 4:
                        item.curve.setPen(mkPen(
                            color=pen.color(), style=pen.style(), width=4))
                    else:
                        self.segment_ledit.clear()
                        self.mean_ledit.clear()
                        self.std_ledit.clear()
                        self.maximum_ledit.clear()
                        self.minimum_ledit.clear()

    def open_joint_dvh(self):
        """."""

        # Get the segments to be displayed
        segments = sorted(tuple(
            set(get_constraint_segments(self.baseline.datahub.segmentation)
                + get_objective_segments(self.baseline.datahub.segmentation))
            &
            set(get_constraint_segments(self.reference.datahub.segmentation)
                + get_objective_segments(self.reference.datahub.segmentation)))
            )

        # Get the colormap
        colors = get_cmap('jet')(linspace(0, 1.0, len(segments)))

        # Create a figure and subplots
        figure, axis = subplots(figsize=(14, 8))

        # Add the dose-volume histogram curve for each segment
        for i, segment in enumerate(segments):

            # Baseline curves
            axis.plot(
                self.baseline.datahub.dose_histogram['evaluation_points'],
                self.baseline.datahub.dose_histogram[segment]['dvh_values'],
                linewidth=1.7,
                color=colors[i],
                linestyle='-',
                label=''.join((segment, ' (baseline) ')))

            # Reference curves
            axis.plot(
                self.reference.datahub.dose_histogram['evaluation_points'],
                self.reference.datahub.dose_histogram[segment]['dvh_values'],
                linewidth=1.7,
                color=colors[i],
                linestyle='--',
                label=''.join((segment, ' (reference) ')))

        # Set x- and y-label
        axis.set_xlabel("Dose per fraction [Gy]", fontsize=16)
        axis.set_ylabel("Relative volume [%]", fontsize=16)

        # Change the tick label sizes for both axes
        axis.tick_params(axis='both', which='major', labelsize=13)

        # Determine the step length on the x-axis
        x_step = min(
            (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100),
            key=lambda x: abs(ceil(max(
                max(
                    self.baseline.datahub.dose_histogram['evaluation_points']),
                max(self.reference.datahub.dose_histogram['evaluation_points'])
                )/x)-20))

        # Set x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(
            int(ceil(max(
                max(
                    self.baseline.datahub.dose_histogram['evaluation_points']),
                max(
                    self.reference.datahub.dose_histogram['evaluation_points'])
                ))/x_step)+1)))
        axis.set_yticks(tuple(i*5 for i in range(21)))

        # Set the x- and y-limits
        axis.set_xlim(left=-0.05)
        axis.set_ylim(-1, 101)

        # Set the facecolor for the axis
        axis.set_facecolor('whitesmoke')

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(which='minor', color='lightgray', linestyle=':',
                  linewidth=0.5)
        axis.minorticks_on()

        # Set the legend and its facecolor
        _, labels = axis.get_legend_handles_labels()
        legend = axis.legend(labels, fontsize=13, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title(
            "pyanno4rt - joint dose-volume histogram (DVH)")

        # Show the plot in screen size
        figure_manager.window.showMaximized()

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
