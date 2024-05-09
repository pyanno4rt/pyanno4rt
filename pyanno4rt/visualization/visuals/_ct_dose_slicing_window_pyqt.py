"""CT/Dose slicing window (PyQt)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from matplotlib.cm import get_cmap, jet, ScalarMappable
from matplotlib.colors import Normalize
from numpy import (nan, nanmax, nanmean, nanmin, nanstd, ndarray, rot90,
                   transpose, unravel_index, zeros)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QScrollBar, QSizePolicy, QVBoxLayout, QWidget)
from pyqtgraph import (GraphicsLayoutWidget, ImageItem, IsocurveItem, mkPen,
                       mkColor, setConfigOptions)
from pyqtgraph.Qt import QtGui

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Set options

setConfigOptions(imageAxisOrder='col-major')

# %% Class definition


class ScrollBar(QWidget):
    """
    Scrollbar class.

    This class provides a scrollbar, including the functional buttons to \
    start/stop the autoplay, reset the view, and go to the first slice, as \
    well as the label field with the changing dose slice statistics.

    Parameters
    ----------
    minimum : int
        Minimum value of the scrollbar.

    maximum : int
        Maximum value of the scrollbar.

    initial_position : int
        Initial position of the scrollbar.

    dose_cube : ndarray
        Cubic array with the dose values.

    Attributes
    ----------
    dose_cube : ndarray
        See 'Parameters'.

    label : string
        Label field with the changing dose slice statistics.

    start_button : object of class `QPushButton`
        Instance of the class `QPushButton`, which represents a functional \
        button to start the autoplay mode.

    pause_button : object of class `QPushButton`
        Instance of the class `QPushButton`, which represents a functional \
        button to pause the autoplay mode.

    reset_button : object of class `QPushButton`
        Instance of the class `QPushButton`, which represents a functional \
        button to reset the visible slice back to the default.

    set_zero_button : object of class `QPushButton`
        Instance of the class `QPushButton`, which represents a functional \
        button to set the visible slice to zero.

    slider : object of class `QScrollBar`
        Instance of the class `QScrollBar`, which creates a slider/scrollbar \
        to navigate through the slices.

    slice_number : int
        Number of the slice for the update of labels and images.
    """

    def __init__(
            self,
            minimum,
            maximum,
            initial_position,
            dose_cube):

        # Call the superclass constructor
        super().__init__()

        # Get the dose cube from the argument
        self.dose_cube = dose_cube

        # Set the vertical layout for the scrollbar
        scrollbar_layout = QVBoxLayout(self)

        # Create a spacing label and add it to the layout
        self.label = QLabel(self)
        self.label.setStyleSheet('''
                                 QLabel
                                     {
                                         color: #FBFAF5;
                                         font-size: 9pt;
                                         max-height: 10%;
                                     }
                                 ''')
        scrollbar_layout.addWidget(self.label)

        # Set the horizontal layout for the scrollbar buttons
        button_layout = QHBoxLayout()

        # Set the button variable names
        button_names = ("start_button", "pause_button", "reset_button",
                        "set_zero_button")

        # Set the button labels
        labels = ("Play", "Pause", "Reset", "Go to zero")

        # Set the stylesheets for the buttons
        button_styles = ('''
                         QPushButton
                             {
                                 background-color: #34A56F;
                             }
                         QPushButton:hover
                             {
                                 background-color: #278664;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #278664;
                             }
                         ''',
                         '''
                         QPushButton
                             {
                                 background-color: #C45C26;
                             }
                         QPushButton:hover
                             {
                                 background-color: #B04812;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #B04812;
                             }
                         ''',
                         '''
                         QPushButton
                             {
                                 background-color: #5CB3FF;
                             }
                         QPushButton:hover
                             {
                                 background-color: #157DEC;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #157DEC;
                             }
                         ''',
                         '''
                         QPushButton
                             {
                                 background-color: #FFAE42;
                             }
                         QPushButton:hover
                             {
                                 background-color: #D7861A;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #D7861A;
                             }
                         ''')

        # Loop over the button elements
        for element in zip(button_names, labels, button_styles):

            # Set the class attribute from the element name
            setattr(self, element[0], QPushButton(element[1]))

            # Get the button attribute
            button = getattr(self, element[0])

            # Set the pointing hand cursor for the button
            button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))

            # Set the stylesheet for the button
            button.setStyleSheet(element[2])

            # Add the button to the layout
            button_layout.addWidget(button)

        # Add the button layout to the scrollbar layout
        scrollbar_layout.addLayout(button_layout)

        # Set the horizontal layout for the slider
        slider_layout = QHBoxLayout()

        # Create the slider and add it to the slider layout
        self.slider = QScrollBar(self)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setSliderPosition(initial_position)
        self.slider.setStyleSheet('''
                                  QScrollBar
                                      {
                                          color: black;
                                          background-color: #FBFAF5;
                                      }
                                 QScrollBar::sub-page
                                     {
                                         background: black;
                                         border: 1px solid #FBFAF5;
                                     }
                                 QScrollBar::add-page
                                     {
                                         background: black;
                                         border: 1px solid #FBFAF5;
                                     }
                                  ''')
        slider_layout.addWidget(self.slider)

        # Add the slider layout to the scrollbar layout
        scrollbar_layout.addLayout(slider_layout)

        # Resize the scrollbar
        self.resize(self.sizeHint())

        # Set the initial label to display dose slice statistics
        self.set_label(self.slider.value())

        # Connect the slider to the event
        self.slider.valueChanged.connect(self.set_label)

    def set_label(
            self,
            slice_number):
        """Set the slice number and the label with the dose statistics."""
        # Set the slice number from the argument
        self.slice_number = slice_number

        # Set the label text with the slice information
        self.label.setText(
            " ".join(("slice",
                      str(self.slice_number+1),
                      "-",
                      "mean dose:",
                      str(round(nanmean(
                          self.dose_cube[:, :, self.slice_number]), 2)),
                      "-",
                      "dose std:",
                      str(round(nanstd(
                          self.dose_cube[:, :, self.slice_number]), 2)),
                      "-",
                      "max dose:",
                      str(round(nanmax(
                          self.dose_cube[:, :, self.slice_number]), 2)),
                      "-",
                      "min dose:",
                      str(round(nanmin(
                          self.dose_cube[:, :, self.slice_number]), 2)))))


class SliceWidget(QWidget):
    """
    Slice widget class.

    This class provides a slice widget, including the widget label, the \
    graphics window showing the sliced array, and the scrollbar with some \
    functionality.

    Parameters
    ----------
    label : string
        Label for the slice widget.

    ct_cube : ndarray
        Cubic array with the CT values.

    dose_cube : ndarray
        Cubic array with the dose values.

    segment_masks : tuple
        Binary masks as indicators for the segment locations.

    segment_colors : tuple
        Colors for the segment contours.

    rotations : int
        Number of 90-degree rotations of the CT/dose cubes.

    Attributes
    ----------
    ct_cube : ndarray
        See 'Parameters'.

    dose_cube : ndarray
        See 'Parameters'.

    dose_cube_with_nan : ndarray
        Cubic array with the dose values, where zeros are replaced with nan.

    segment_masks : tuple
        See 'Parameters'.

    ct_image : object of class `ImageItem`
        Instance of the class `ImageItem`, which displays the CT image.

    dose_image : object of class `ImageItem`
        Instance of the class `ImageItem`, which displays the dose image.

    dose_contours : list
        Isodose contours for different levels.

    segment_contours : list
        Isolevel contours for different segments.

    scrollbar : object of class `ScrollBar`
        Instance of the class `ScrollBar`, which enables scrolling through \
        the image slices with a horizontal bar-style element.

    initial_position : int
        Initial position of the scrollbar.

    timer : object of class `QTimer`
        Instance of the class `QTimer`, which periodically performs slice \
        increments to allow for the autoplay mode.

    current_position : int
        Current position of the scrollbar.
    """

    def __init__(
            self,
            label,
            ct_cube,
            dose_cube,
            segment_masks,
            segment_colors,
            rotations):

        # Call the superclass constructor
        super().__init__()

        def add_label(layout, label):
            """Create and add the label above the slice window."""
            label = QLabel(label)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet('''
                                QLabel
                                    {
                                        color: #FBFAF5;
                                        font-size: 14pt;
                                        max-height: 25%;
                                    }
                                ''')
            layout.addWidget(label)

        # Rotate the CT, dose, and segmentation cube
        self.ct_cube = rot90(ct_cube, rotations)
        self.dose_cube = rot90(dose_cube, rotations)
        self.segment_masks = tuple(rot90(mask, rotations)
                                   for mask in segment_masks)

        # Set the zero dose and zero segmentation values to nan
        self.dose_cube_with_nan = self.dose_cube.copy()
        self.dose_cube_with_nan[self.dose_cube_with_nan == 0] = nan

        # Set the vertical layout for the slice widget
        slice_layout = QVBoxLayout(self)

        # Add the label to the slice layout
        add_label(slice_layout, label)

        # Create an image window, set its size, and add it to the slice layout
        image_window = GraphicsLayoutWidget(
            title="pyanno4rt - CT/dose slice plot")
        image_window.setMaximumHeight(int(self.frameGeometry().height()*1.4))
        slice_layout.addWidget(image_window)

        # Add the view box to the image window
        viewbox = image_window.addViewBox()

        # Initialize the CT image and set the borders
        self.ct_image = ImageItem()
        self.ct_image.setBorder('#FBFAF5')

        # Initialize the dose image and set the borders/opacity
        self.dose_image = ImageItem()
        self.dose_image.setBorder('#FBFAF5')
        self.dose_image.setOpacity(0.7)

        # Get and initialize the colormap for the dose image
        colormap_dose = get_cmap('jet')
        colormap_dose._init()

        # Set the lookup table for the dose image
        self.dose_image.setLookupTable((colormap_dose._lut*255).view(ndarray))

        # Initialize the segmentation images (1 per segment mask)
        segment_images = [ImageItem() for _ in self.segment_masks]

        # Add the image items to the viewbox
        viewbox.addItem(self.ct_image)
        viewbox.addItem(self.dose_image)
        for _, image in enumerate(segment_images):
            viewbox.addItem(image)

        # Get the dose quantiles
        quantiles = [0.1*factor1 for factor1 in range(1, 10)]
        quantiles.extend([0.95+0.05*factor2 for factor2 in range(0, 6)])

        # Determine the reference dose value
        reference_dose = self.dose_cube.max()/1.2

        # Compute the dose levels for the contours
        levels = [reference_dose*level for level in quantiles]

        # Normalize the dose levels to the unit interval
        norm = Normalize(vmin=min(levels), vmax=max(levels), clip=True)

        # Create a mapping between normalized values and colors
        mapper = ScalarMappable(norm=norm, cmap=jet)

        # Initialize the dose contours
        self.dose_contours = []

        # Loop over the dose levels
        for level in levels:

            # Initialize the contour for the specific level
            contour = IsocurveItem(level=level, pen=mkPen(
                tuple([255*rgba for rgba in mapper.to_rgba(level)]),
                width=2.5))

            # Bind the contour to the dose image
            contour.setParentItem(self.dose_image)

            # Set the z-value for the contour
            contour.setZValue(5)

            # Append the contour to the dose contours
            self.dose_contours.append(contour)

        # Initialize the segment contours
        self.segment_contours = []

        # Loop over the segment images
        for color, image in zip(segment_colors, segment_images):

            # Initialize the contour
            contour = IsocurveItem(level=1, pen=mkPen(mkColor(color),
                                                      width=2.5))

            # Bind the contour to the segment image
            contour.setParentItem(image)

            # Set the z-value for the contour
            contour.setZValue(5)

            # Append the contour to the segment contours
            self.segment_contours.append(contour)

        # Initialize the scrollbar and add it to the slice layout
        self.scrollbar = ScrollBar(
            minimum=0,
            maximum=self.ct_cube.shape[2]-1,
            initial_position=int(self.ct_cube.shape[2]/2),
            dose_cube=self.dose_cube_with_nan)
        slice_layout.addWidget(self.scrollbar)

        # Get the initial scroll position
        self.initial_position = self.scrollbar.slider.value()

        # Initialize the current scroll position
        self.current_position = None

        # Initialize the images by updating to the initial position
        self.update_images()

        # Initialize the timer
        self.timer = QTimer()

        # Connect timer and scrollbar to the events
        self.timer.timeout.connect(self.increment_slice)
        self.scrollbar.start_button.clicked.connect(self.start_autoplay)
        self.scrollbar.pause_button.clicked.connect(self.end_autoplay)
        self.scrollbar.reset_button.clicked.connect(self.reset_scroll_position)
        self.scrollbar.set_zero_button.clicked.connect(self.set_scroll_to_zero)
        self.scrollbar.slider.valueChanged.connect(self.update_images)

    def update_images(self):
        """Update the images when scrolling."""
        # Update the CT image
        self.ct_image.setImage(
            self.ct_cube[:, :, self.scrollbar.slice_number])

        # Update the dose image
        self.dose_image.setImage(
            self.dose_cube_with_nan[:, :, self.scrollbar.slice_number])

        # Loop over the dose contours
        for contour in self.dose_contours:

            # Update the dose contour lines
            contour.setData(self.dose_cube[:, :, self.scrollbar.slice_number])

        # Loop over the segment contours
        for mask, contour in zip(self.segment_masks, self.segment_contours):

            # Update the segment contour lines
            contour.setData(mask[:, :, self.scrollbar.slice_number])

    def increment_slice(self):
        """Increment the slice number."""
        # Check if the current position already equals the last slice
        if self.current_position == self.ct_cube.shape[2]-1:

            # End the autoplay mode
            self.end_autoplay()

        else:

            # Increment the current position by one
            self.current_position += 1

            # Set the scrollbar to the new position
            self.scrollbar.slider.setSliderPosition(self.current_position)

    def start_autoplay(self):
        """Start the autoplay mode."""
        # Get the current position from the scrollbar
        self.current_position = self.scrollbar.slider.value()

        # Set the timer step length to 500 ms
        self.timer.start(500)

        # Enable/disable the functional buttons
        self.scrollbar.start_button.setEnabled(False)
        self.scrollbar.pause_button.setEnabled(True)
        self.scrollbar.reset_button.setEnabled(False)
        self.scrollbar.set_zero_button.setEnabled(False)

    def end_autoplay(self):
        """End the autoplay mode."""
        # Stop the timer
        self.timer.stop()

        # Enable/disable the functional buttons
        self.scrollbar.start_button.setEnabled(True)
        self.scrollbar.pause_button.setEnabled(False)
        self.scrollbar.reset_button.setEnabled(True)
        self.scrollbar.set_zero_button.setEnabled(True)

    def reset_scroll_position(self):
        """Reset the scrollbar to the initial position."""
        # Set the scrollbar to the initial position
        self.scrollbar.slider.setSliderPosition(self.initial_position)

        # Enable all functional buttons
        self.scrollbar.start_button.setEnabled(True)
        self.scrollbar.pause_button.setEnabled(True)
        self.scrollbar.reset_button.setEnabled(True)
        self.scrollbar.set_zero_button.setEnabled(True)

    def set_scroll_to_zero(self):
        """Set the scrollbar to the zero position."""
        self.scrollbar.slider.setSliderPosition(0)


class CtDoseSlicingWindowPyQt(QMainWindow):
    """
    CT/Dose slicing window (PyQt) class.

    This class provides an interactive plot of the patient's CT/dose slices \
    on the axial, sagittal and coronal axes, including the segment contours, \
    dose level curves, and a scrolling and autoplay functionality.

    Attributes
    ----------
    category : string
        Plot category for assignment to the button groups in the visual \
        interface.

    name : string
        Attribute name of the classes' instance in the visual interface.

    label : string
        Label of the plot button in the visual interface.
    """

    # Set the class attributes for the visual interface integration
    category = "Treatment plan evaluation"
    name = "ct_dose_plotter"
    label = "CT/Dose slice plot"

    def view(self):
        """Open the full-screen view on the CT/dose slicing window."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening CT/dose slice plot ...")

        # Get the CT, segmentation and optimized dose data from the datahub
        computed_tomography = hub.computed_tomography
        segmentation = hub.segmentation
        optimized_dose = hub.optimization['optimized_dose'].copy()

        def add_logo(layout):
            """Create and add the pyanno4rt logo."""
            logo = QLabel(self)
            pixmap = QtGui.QPixmap('./logo/logo_white_512.png')
            pixmap = pixmap.scaled(int(pixmap.width()/2),
                                   int(pixmap.height()/2))
            logo.setPixmap(pixmap)
            logo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet('''
                               QLabel
                                   {
                                       margin: 0 0 0 0;
                                   }
                               ''')
            layout.addWidget(logo)

        def generate_segment_mask(segment):
            """Generate the segmentation masks as a single cube."""
            # Initialize the segment mask
            segment_mask = zeros(computed_tomography['cube_dimensions'])

            # Insert ones at the indices of the segment
            segment_mask[unravel_index(
                hub.segmentation[segment]['raw_indices'],
                hub.computed_tomography['cube_dimensions'], order='F')] = 1

            return segment_mask

        def generate_view(layout, label, orientation, segment_masks,
                          rotations):
            """Create a slice widget and add it to the layout."""
            slice_view = SliceWidget(
                label=label,
                ct_cube=transpose(computed_tomography['cube'], orientation),
                dose_cube=transpose(optimized_dose, orientation),
                segment_masks=tuple(transpose(mask, orientation)
                                    for mask in segment_masks),
                segment_colors=tuple(
                    255*segmentation[segment]['parameters']['visibleColor']
                    for segment in (*segmentation,)),
                rotations=rotations)
            layout.addWidget(slice_view)

        # Set the window title
        self.setWindowTitle('pyanno4rt - CT/dose slicing')

        # Set the window style sheet
        self.setStyleSheet('background-color: black;')

        # Initialize the central widget and add it to the window
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set the vertical layout for the central widget
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        # Add the logo to the central layout
        add_logo(central_layout)

        # Set the horizontal layout for the slice axes views
        view_layout = QHBoxLayout()

        # Generate the segmentation cube
        segment_masks = tuple(generate_segment_mask(segment)
                              for segment in (*segmentation,))

        # Create the slicing widgets and add them to the layout
        generate_view(view_layout, "Axial", (0, 1, 2), segment_masks, 3)
        generate_view(view_layout, "Sagittal", (0, 2, 1), segment_masks, 0)
        generate_view(view_layout, "Coronal", (2, 1, 0), segment_masks, 1)

        # Add the axes views to the central layout
        central_layout.addLayout(view_layout)

        # Show the plot in screen size
        self.showMaximized()
