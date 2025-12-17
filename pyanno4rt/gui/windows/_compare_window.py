"""Plan comparison window."""

# Author: Tim Ortkamp

# %% External package import

from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import ceil, linspace, multiply
from PyQt5.QtWidgets import QMainWindow
from pyqtgraph import mkPen

# %% Internal package import

from pyanno4rt.evaluation import DVH
from pyanno4rt.gui._custom_styles import cbox, sbox, pbutton_composer
from pyanno4rt.gui.compilations.compare_window import Ui_compare_window
from pyanno4rt.gui.custom_widgets import (
    DVHGraphCompareWidget, SliceCompareWidget)
from pyanno4rt.tools import (
    get_all_segments, get_machine_learning_components,
    get_radiobiological_components)

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
        self.baseline_dvh_widget = DVHGraphCompareWidget(self)
        self.reference_dvh_widget = DVHGraphCompareWidget(self)
        self.difference_dvh_widget = DVHGraphCompareWidget(self)

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

        # Set the stylesheets
        self.set_styles({
            'plane_cbox': cbox,
            'opacity_sbox': sbox,
            'joint_dvh_pbutton': pbutton_composer,
            'joint_outcome_pbutton': pbutton_composer,
            'close_compare_pbutton': pbutton_composer})

        # Loop over the QComboBox and QSpinBox elements
        for box in ('plane_cbox', 'opacity_sbox'):

            # Install the custom event filter
            getattr(self, box).installEventFilter(parent)

        # Set the view box links
        self.set_links()

        # Connect the fields with the event signals
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

    def set_links(self):
        """."""

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
        self.baseline_dvh_widget.plot_widget.getPlotItem().vb.setXLink(
            self.reference_dvh_widget.plot_widget.getPlotItem().vb)
        self.baseline_dvh_widget.plot_widget.getPlotItem().vb.setYLink(
            self.reference_dvh_widget.plot_widget.getPlotItem().vb)

    def connect_signals(self):
        """Connect the fields with the event signals."""

        #
        self.plane_cbox.currentTextChanged.connect(
            self.adjust_slider_by_orientation)
        self.plane_cbox.currentTextChanged.connect(
            self.baseline_dose_slice_widget.change_orientation)
        self.plane_cbox.currentTextChanged.connect(
            self.reference_dose_slice_widget.change_orientation)
        self.plane_cbox.currentTextChanged.connect(
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
        self.joint_outcome_pbutton.clicked.connect(self.open_joint_outcome)

        #
        self.close_compare_pbutton.clicked.connect(self.close)

    def add_plans(self, baseline, reference):
        """."""

        #
        self.baseline, self.reference = baseline, reference

        #
        self.baseline_dose_slice_widget.reset_images()
        self.reference_dose_slice_widget.reset_images()
        self.difference_dose_slice_widget.reset_images()

        # Get the joint minimum and maximum dose
        minimum = min(
            baseline.fluence_optimizer.optimized_dose.min(),
            reference.fluence_optimizer.optimized_dose.min())
        maximum = max(
            baseline.fluence_optimizer.optimized_dose.max(),
            reference.fluence_optimizer.optimized_dose.max())

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
        baseline_dvh = baseline.dvh.histogram
        reference_dvh = reference.dvh.histogram

        #
        dvh = DVH(dvh_type='cumulative', number_of_points=1000)
        dvh.evaluate_segments(
            baseline.patient_handler.segmentation,
            baseline.fluence_optimizer.optimized_dose
            - reference.fluence_optimizer.optimized_dose)
        dvh_diff = dvh.histogram

        #
        joint_segments = sorted(tuple(
            set(key for key in baseline_dvh if key != 'evaluation_points')
            & set(key for key in reference_dvh if key != 'evaluation_points')),
            key=lambda t: t[0])

        #
        x_range = (
            0, max(baseline_dvh['evaluation_points'][-1],
                   reference_dvh['evaluation_points'][-1]))

        # Add the style and input data to the DVH widget
        self.baseline_dvh_widget.add_style_and_data(
            baseline_dvh, x_range=x_range, baseline=baseline)
        self.reference_dvh_widget.add_style_and_data(
            reference_dvh, x_range=x_range, reference=reference)
        self.difference_dvh_widget.add_style_and_data(
            dvh_diff, baseline=baseline, reference=reference)

        # Update the plot of the DVH widget
        self.baseline_dvh_widget.update_dvh(joint_segments)
        self.reference_dvh_widget.update_dvh(joint_segments)
        self.difference_dvh_widget.update_dvh(joint_segments)

        # Check if any of the plans has no ML components
        if any(len(
                get_machine_learning_components(components)
                + get_radiobiological_components(components)) == 0
                for components in (
                        self.baseline.plan_handler.components,
                        self.reference.plan_handler.components)):

            # Disable the button
            self.joint_outcome_pbutton.setEnabled(False)

        else:

            # Enable the button
            self.joint_outcome_pbutton.setEnabled(True)

    def set_titles(
            self,
            baseline_text,
            reference_text,
            difference_text):
        """."""

        #
        self.baseline_label.setText(baseline_text)
        self.reference_label.setText(reference_text)
        self.difference_label.setText(difference_text)

    def adjust_slider_by_orientation(self):
        """Adjust the slider for slice selection by the orientation."""

        # Create a mapping between planes and axes
        mapping = {'axial': 2, 'coronal': 0, 'sagittal': 1}

        # Get the depth of the current plane
        plane_depth = self.baseline.patient_handler.computed_tomography[
            'cube_dimensions'][mapping[self.plane_cbox.currentText()]]

        # Set the range of the slice selection scrollbar
        self.slice_selection_sbar.setRange(0, plane_depth-1)

        # Set the initial scrollbar value
        self.slice_selection_sbar.setValue(int((plane_depth-1)/2))

    def select_dvh_curves(self, event):
        """."""

        #
        triples = tuple(zip(*(
            widget.plot_widget.getPlotItem().listDataItems()
            for widget in (
                    self.baseline_dvh_widget, self.reference_dvh_widget,
                    self.difference_dvh_widget))))

        #
        for triple in triples:

            #
            if any(item.curve == event or item == event for item in triple):

                #
                clicked = next(
                    item for item in triple
                    if item.curve == event or item == event)

                #
                linked = [
                    item for item in triple
                    if item.curve != event and item != event]

                #
                cpen = clicked.curve.opts['pen']

                #
                if cpen.width() != 5:

                    #
                    clicked.curve.setPen(mkPen(
                        color=cpen.color(), style=cpen.style(), width=5))

                    #
                    for link in linked:

                        #
                        lpen = link.curve.opts['pen']

                        #
                        link.curve.setPen(mkPen(
                            color=lpen.color(), style=lpen.style(), width=3))

                else:

                    #
                    clicked.curve.setPen(mkPen(
                        color=cpen.color(), style=cpen.style(), width=2))

                    #
                    self.segment_ledit.clear()
                    self.mean_ledit.clear()
                    self.std_ledit.clear()
                    self.maximum_ledit.clear()
                    self.minimum_ledit.clear()

                    #
                    for link in linked:

                        #
                        lpen = link.curve.opts['pen']

                        #
                        link.curve.setPen(mkPen(
                            color=lpen.color(), style=lpen.style(), width=2))

            else:

                #
                for item in triple:

                    #
                    pen = item.curve.opts['pen']

                    #
                    item.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=2))

    def unselect_dvh_curves(self, event):
        """."""

        #
        if not event.isAccepted():

            for widget in (self.baseline_dvh_widget, self.reference_dvh_widget,
                           self.difference_dvh_widget):

                # Get all plot items
                items = widget.plot_widget.getPlotItem().listDataItems()

                for item in items:
                    pen = item.curve.opts['pen']
                    item.curve.setPen(mkPen(color=pen.color(),
                                            style=pen.style(),
                                            width=2))

                self.segment_ledit.clear()
                self.mean_ledit.clear()
                self.std_ledit.clear()
                self.maximum_ledit.clear()
                self.minimum_ledit.clear()

    def open_joint_dvh(self):
        """Open the joint DVH graph."""

        # Get the components
        baseline_components = self.baseline.plan_handler.components
        reference_components = self.reference.plan_handler.components

        # Get the dose histogram dictionaries
        baseline_dvh = self.baseline.dvh.histogram
        reference_dvh = self.reference.dvh.histogram

        # Get the segments to be displayed
        segments = sorted(
            set(get_all_segments(baseline_components)) &
            set(get_all_segments(reference_components)))

        # Get the colormap
        colors = get_cmap('tab20b')(linspace(0, 1.0, len(segments)))

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Loop over the segments
        for index, segment in enumerate(segments):

            # Plot the baseline DVH curves
            axis.plot(
                baseline_dvh['evaluation_points'],
                100*baseline_dvh[segment],
                linewidth=1.5,
                color=colors[index],
                linestyle='-',
                label=f'{segment} (baseline) ')

            # Plot the reference DVH curves
            axis.plot(
                reference_dvh['evaluation_points'],
                100*reference_dvh[segment],
                linewidth=1.5,
                color=colors[index],
                linestyle='--',
                label=f'{segment} (reference) ')

        # Set the x- and y-labels
        axis.set_xlabel("Dose [Gy]", fontsize=16)
        axis.set_ylabel("Relative volume [%]", fontsize=16)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=13)

        # Determine the step length on the x-axis
        x_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
            key=lambda x: abs(ceil(max(
                baseline_dvh['evaluation_points'][-1],
                reference_dvh['evaluation_points'][-1])/x)-20))

        # Set the x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(int(ceil(max(
                baseline_dvh['evaluation_points'][-1],
                reference_dvh['evaluation_points'][-1]))/x_step)+1)))
        axis.set_yticks(tuple(i*5 for i in range(21)))

        # Set the x- and y-limits
        axis.set_xlim(left=-0.05)
        axis.set_ylim(-1, 101)

        # Set the facecolor for the axis
        axis.set_facecolor('whitesmoke')

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(
            which='minor', color='lightgray', linestyle=':', linewidth=0.5)
        axis.minorticks_on()

        # Configure the legend
        _, labels = axis.get_legend_handles_labels()
        legend = axis.legend(labels, fontsize=13, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - joint DVH graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()

    def open_joint_outcome(self):
        """Open the joint iterative outcome graph."""

        # Get the optimizers
        baseline_opt = self.baseline.fluence_optimizer
        reference_opt = self.reference.fluence_optimizer

        # Get the outcome model-based optimization components
        baseline_components, reference_components = ((
            get_machine_learning_components(components)
            + get_radiobiological_components(components))
            for components in (
                    self.baseline.plan_handler.components,
                    self.reference.plan_handler.components))

        # Get the baseline tracks to be displayed
        baseline_tracker = {
            component.track_id: component.translate(list(
                baseline_opt.problem.tracker[component.track_id]))
            for component in baseline_components}

        # Get the reference tracks to be displayed
        reference_tracker = {
            component.track_id: component.translate(list(
                reference_opt.problem.tracker[component.track_id]))
            for component in reference_components}

        # Get the track statistics
        track_len = max(
            len(track) for tracker in (baseline_tracker, reference_tracker)
            for track in tracker.values())
        track_num = len(baseline_tracker) + len(reference_tracker)

        # Set the colormap
        colors = get_cmap('tab20b')(linspace(0, 1.0, track_num))

        # Determine the step length on the x-axis
        x_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(6)),
            key=lambda x: abs(ceil(track_len/x)-20))

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Set the plot title
        axis.set_title(label='', fontsize=16)

        # Loop over the tracks
        for index, (track, values) in enumerate(baseline_tracker.items()):

            # Plot the track
            axis.plot(
                range(1, len(values)+1),
                multiply(values, 100),
                color=colors[index],
                linestyle='-',
                linewidth=1.5,
                label=f'{track} (baseline) ')

        # Loop over the reference tracks
        for index, (track, values) in enumerate(reference_tracker.items()):

            # Plot the track
            axis.plot(
                range(1, len(values)+1),
                multiply(values, 100),
                color=colors[len(baseline_tracker)+index],
                linestyle='--',
                linewidth=1.5,
                label=f'{track} (reference) ')

        # Set the x- and y-labels
        axis.set_xlabel(xlabel="Evaluation step", fontsize=16)
        axis.set_ylabel(ylabel="Outcome value [%]", fontsize=16)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=13)

        # Set the x- and y-ticks
        axis.set_xticks(tuple(
            i*x_step for i in range(int(ceil(track_len/x_step))+1)))
        axis.set_yticks(tuple(100*i/20 for i in range(21)))

        # Set the x- and y-limits
        axis.set_xlim(0, track_len+x_step/2)
        axis.set_ylim(-1, 101)

        # Set the facecolor for the axis
        axis.set_facecolor('whitesmoke')

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(
            which='minor', color='lightgray', linestyle=':', linewidth=0.5)
        axis.minorticks_on()

        # Configure the legend
        _, labels = axis.get_legend_handles_labels()
        legend = axis.legend(labels, fontsize=13, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("Iterative outcome graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()

    def position(self):
        """Set the window position."""

        # Reset the window size
        self.resize(self.parent.size())

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the plan comparison window."""

        # Hide the window
        self.hide()
