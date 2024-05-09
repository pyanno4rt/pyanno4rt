"""Slice widget."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import (nan, ndarray, rot90, transpose, unravel_index, zeros)
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (GraphicsLayoutWidget, ImageItem, IsocurveItem, mkColor,
                       mkPen)

# %% Class definition


class SliceWidget(QWidget):
    """."""

    def __init__(self, parent=None):

        # Call the superclass constructor
        super().__init__()

        # 
        self.parent = parent

        # Set the vertical layout for the slice widget
        slice_layout = QVBoxLayout(self)
        slice_layout.setContentsMargins(10, 10, 10, 0)

        # Create an image window, set its size, and add it to the slice layout
        image_window = GraphicsLayoutWidget()
        slice_layout.addWidget(image_window)

        # Add the view box to the image window
        self.viewbox = image_window.addViewBox()

        # 
        self.ct_image = ImageItem()
        self.viewbox.addItem(self.ct_image)

        # 
        self.dose_image = ImageItem()
        self.dose_image.setOpacity(0.7)
        colormap_dose = colormaps['jet']
        colormap_dose._init()
        self.dose_image.setLookupTable((colormap_dose._lut*255).view(ndarray))
        self.viewbox.addItem(self.dose_image)

        # 
        self.slice = None

        # 
        self.positions = None

        # 
        self.ct_cube = None
        self.dose_cube = None
        self.dose_cube_with_nan = None
        self.dose_contours = None
        self.segment_masks = None
        self.segment_contours = None

    def add_ct(self, ct_cube):
        """."""

        # 
        self.ct_cube = rot90(transpose(ct_cube, (0, 1, 2)), 3)

        # 
        self.positions = self.parent.plans[
            self.parent.plan_ledit.text()].datahub.computed_tomography['z']

    def add_dose(self, dose_cube):

        self.dose_cube = rot90(transpose(dose_cube, (0, 1, 2)), 3)

        self.dose_cube_with_nan = self.dose_cube.copy()
        self.dose_cube_with_nan[self.dose_cube_with_nan == 0] = nan

        quantiles = [0.1*factor1 for factor1 in range(1, 10)]
        quantiles.extend([0.95+0.05*factor2 for factor2 in range(0, 6)])

        reference_dose = self.dose_cube.max()/1.2

        levels = [reference_dose*level for level in quantiles]
        norm = Normalize(vmin=min(levels), vmax=max(levels), clip=True)
        mapper = ScalarMappable(norm=norm, cmap=colormaps['jet'])

        self.dose_contours = []
        for level in levels:
            contour = IsocurveItem(level=level, pen=mkPen(
                tuple([255*rgba for rgba in mapper.to_rgba(level)]),
                width=2.5))
            contour.setParentItem(self.dose_image)
            contour.setZValue(5)
            self.dose_contours.append(contour)

    def add_segments(self, computed_tomography, segmentation):

        def generate_segment_mask(segment):
            """Generate the segmentation masks as a single cube."""
            # Initialize the segment mask
            segment_mask = zeros(computed_tomography['cube_dimensions'])

            # Insert ones at the indices of the segment
            segment_mask[unravel_index(
                segmentation[segment]['raw_indices'],
                computed_tomography['cube_dimensions'], order='F')] = 1

            return segment_mask

        segment_colors = tuple(
            255*segmentation[segment]['parameters']['visibleColor']
            for segment in (*segmentation,))

        raw_masks = tuple(
            generate_segment_mask(segment) for segment in (*segmentation,))

        segment_images = [ImageItem() for _ in raw_masks]
        for image in segment_images:
            self.viewbox.addItem(image)

        self.segment_masks = tuple(rot90(
            transpose(mask, (0, 1, 2)), 3) for mask in raw_masks)

        self.segment_contours = []
        for color, image in zip(segment_colors, segment_images):
            contour = IsocurveItem(level=1, pen=mkPen(mkColor(color),
                                                      width=2.5))
            contour.setParentItem(image)
            contour.setZValue(5)
            self.segment_contours.append(contour)

    def change_dose_opacity(self):
        """."""

        # 
        self.dose_image.setOpacity(self.parent.opacity_sbox.value()/100)

        # 
        self.update_images()

    def change_image_slice(self):
        """."""

        # 
        self.slice = self.parent.slice_selection_sbar.value()

        # 
        self.update_images()

    def reset_images(self):
        """."""

        if self.ct_cube is not None:

            # Update the CT image
            self.ct_image.clear()
            self.ct_cube = None

        if self.dose_cube_with_nan is not None:

            # Update the dose image
            self.dose_image.clear()
            self.dose_cube_with_nan = None

        if self.dose_cube is not None or self.dose_contours is not None:

            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(zeros(self.dose_cube[:, :, self.slice].shape))

            self.dose_cube = None
            self.dose_contours = None

        if (self.segment_masks is not None
                or self.segment_contours is not None):

            # Loop over the segment contours
            for mask, contour in zip(self.segment_masks,
                                     self.segment_contours):

                # Update the segment contour lines
                contour.setData(zeros(mask[:, :, self.slice].shape))

            self.segment_masks = None
            self.segment_contours = None

        self.parent.slice_selection_pos.clear()

    def update_images(self):
        """Update the images when scrolling."""

        if self.ct_cube is not None:

            # Update the CT image
            self.ct_image.setImage(self.ct_cube[:, :, self.slice])

        if self.dose_cube_with_nan is not None:

            # Update the dose image
            self.dose_image.setImage(self.dose_cube_with_nan[:, :, self.slice])

        if self.dose_cube is not None and self.dose_contours is not None:

            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(self.dose_cube[:, :, self.slice])

        if (self.segment_masks is not None
                and self.segment_contours is not None):

            # Loop over the segment contours
            for mask, contour in zip(self.segment_masks,
                                     self.segment_contours):

                # Update the segment contour lines
                contour.setData(mask[:, :, self.slice])

        # 
        if self.positions is not None:

            self.parent.slice_selection_pos.setText(
                ''.join(('z = ', str(self.positions[self.slice]), ' mm')))
