"""Slice widget."""

# Author: Tim Ortkamp

# %% External package import

from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import (nan, rot90, transpose, unravel_index, zeros)
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (
    colormap, ColorBarItem, GraphicsLayoutWidget, ImageItem, IsocurveItem,
    mkColor, mkPen)

# %% Class definition


class SliceWidget(QWidget):
    """."""

    def __init__(self, parent=None, cmap='jet'):

        # Call the superclass constructor
        super().__init__()

        # 
        self.parent = parent
        self.cmap = cmap

        # Set the vertical layout for the slice widget
        slice_layout = QVBoxLayout(self)

        # Create an image window, set its size, and add it to the slice layout
        self.image_window = GraphicsLayoutWidget()
        slice_layout.addWidget(self.image_window)

        # Add the view box to the image window
        self.viewbox = self.image_window.addViewBox()

        # 
        self.ct_image = ImageItem()
        self.viewbox.addItem(self.ct_image)

        # 
        self.dose_image = ImageItem()
        self.dose_image.setOpacity(0.7)
        self.dose_image.setLookupTable(
            colormap.get(cmap, 'matplotlib').getLookupTable(0.0, 1.0))
        self.viewbox.addItem(self.dose_image)

        # 
        self.bar = ColorBarItem(
            interactive=False, width=25, label='',
            rounding=0.1, colorMap=colormap.get(cmap, 'matplotlib'),
            orientation='vertical')
        self.bar.setImageItem(self.dose_image)

        # 
        self.slice = None
        self.positions = None

        # 
        self.ct_cube = None
        self.dose_cube = None
        self.dose_cube_with_nan = None
        self.dose_contours = None
        self.segment_masks = None
        self.segment_contours = None

        # 
        self.orientations = {
            'axial': ((0, 1, 2), 3, 'z'),
            'coronal': ((0, 2, 1), 0, 'x'),
            'sagittal': ((2, 1, 0), 1, 'y')}

        # 
        self.parent.plane_cbox.currentTextChanged.connect(
            self.parent.adjust_slider_by_orientation)

    def add_ct(self):
        """."""

        def generate_segment_mask(segment):
            """Generate the segmentation masks as a single cube."""
            # Initialize the segment mask
            segment_mask = zeros(computed_tomography['cube_dimensions'])

            # Insert ones at the indices of the segment
            segment_mask[unravel_index(
                segmentation[segment]['raw_indices'],
                computed_tomography['cube_dimensions'], order='F')] = 1

            return segment_mask

        # 
        self.plan = self.parent.plans[self.parent.plan_ledit.text()]

        # 
        computed_tomography = self.plan.datahub.computed_tomography
        segmentation = self.plan.datahub.segmentation

        # 
        self.ct_cube = self.plan.datahub.computed_tomography['cubeHU']

        # 
        self.positions = (
            computed_tomography['x'],
            computed_tomography['y'],
            computed_tomography['z'])

        # 
        self.segment_masks = tuple(
            generate_segment_mask(segment) for segment in segmentation)

        segment_colors = tuple(
            255*segmentation[segment]['parameters']['visibleColor']
            for segment in (*segmentation,))

        segment_images = [ImageItem() for _ in self.segment_masks]
        for image in segment_images:
            self.viewbox.addItem(image)

        self.segment_contours = []
        for color, image in zip(segment_colors, segment_images):
            contour = IsocurveItem(level=1, pen=mkPen(mkColor(color),
                                                      width=2.5))
            contour.setParentItem(image)
            contour.setZValue(5)
            self.segment_contours.append(contour)

    def add_dose(self):
        """."""

        # 
        self.plan = self.parent.plans[self.parent.plan_ledit.text()]

        # 
        self.dose_cube = self.plan.datahub.optimization['optimized_dose']
        self.minimum, self.maximum = self.dose_cube.min(), self.dose_cube.max()

        # 
        self.dose_cube_with_nan = self.dose_cube.copy()
        self.dose_cube_with_nan[self.dose_cube_with_nan == 0] = nan

        quantiles = [0.1*factor1 for factor1 in range(1, 10)]
        quantiles.extend([0.95, 0.975, 0.99, 0.999])
        levels = [self.maximum*level for level in quantiles]
        norm = Normalize(vmin=min(levels), vmax=max(levels), clip=True)
        mapper = ScalarMappable(norm=norm, cmap=colormaps[self.cmap])

        self.dose_contours = []
        for level in levels:
            contour = IsocurveItem(level=level, pen=mkPen(
                tuple([255*rgba for rgba in mapper.to_rgba(level)]),
                width=2.5))
            contour.setParentItem(self.dose_image)
            contour.setZValue(5)
            self.dose_contours.append(contour)

    def add_image_data(self):
        """."""

        # 
        self.add_ct()
        self.add_dose()

    def update_ct(self):
        """."""

        # 
        orientation, rotations, _ = self.orientations[
            self.parent.plane_cbox.currentText()]

        # 
        if self.ct_cube is not None:

            # 
            ct_cube = rot90(
                transpose(self.ct_cube, orientation), rotations)

            # Update the CT image
            self.ct_image.setImage(ct_cube[:, :, self.slice])

    def update_dose(self):
        """."""

        # 
        orientation, rotations, _ = self.orientations[
            self.parent.plane_cbox.currentText()]

        if self.dose_cube_with_nan is not None:

            # 
            dose_cube_with_nan = rot90(
                transpose(self.dose_cube_with_nan, orientation), rotations)

            # Update the dose image
            self.dose_image.setImage(dose_cube_with_nan[:, :, self.slice])

            # 
            self.image_window.addItem(self.bar)

            # 
            self.bar.setLevels((min(0, round(self.minimum, 1)-0.1),
                                round(self.maximum, 1)+0.1))

    def update_dose_contours(self):
        """."""

        # 
        orientation, rotations, _ = self.orientations[
            self.parent.plane_cbox.currentText()]

        if self.dose_cube is not None and self.dose_contours is not None:

            # 
            dose_cube = rot90(
                transpose(self.dose_cube, orientation), rotations)

            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(dose_cube[:, :, self.slice])

    def update_segment_contours(self):
        """."""

        # 
        orientation, rotations, _ = self.orientations[
            self.parent.plane_cbox.currentText()]

        if (self.segment_masks is not None
                and self.segment_contours is not None):

            # 
            segment_masks = tuple(rot90(
                transpose(mask, orientation), rotations)
                for mask in self.segment_masks)

            # Loop over the segment contours
            for mask, contour in zip(segment_masks, self.segment_contours):

                # Update the segment contour lines
                contour.setData(mask[:, :, self.slice])

    def update_parent(self):
        """."""

        # 
        axis = self.orientations[self.parent.plane_cbox.currentText()][2]

        # 
        if self.positions is not None:

            # 
            position = round(
                self.plan.datahub.computed_tomography[axis][self.slice], 2)

            # 
            self.parent.slice_selection_pos.setText(
                f'{axis} = {position} mm')

        self.parent.set_enabled((
            'plane_cbox', 'opacity_sbox', 'slice_selection_sbar'))

    def update_images(self):
        """Update all images."""

        self.update_ct()
        self.update_dose()
        self.update_dose_contours()
        self.update_segment_contours()
        self.update_parent()

    def reset_ct(self):
        """."""

        if self.ct_cube is not None:

            # Update the CT image
            self.ct_image.clear()
            self.ct_cube = None

    def reset_dose(self):
        """."""

        if self.dose_cube_with_nan is not None:

            # Update the dose image
            self.dose_image.clear()
            self.dose_cube_with_nan = None

            # 
            try:

                # 
                self.image_window.removeItem(self.bar)

            except ValueError:

                # 
                pass

    def reset_dose_contours(self):
        """."""

        # 
        orientation, rotations, axis = self.orientations[
            self.parent.plane_cbox.currentText()]

        if self.dose_cube is not None and self.dose_contours is not None:

            # 
            dose_cube = rot90(
                transpose(self.dose_cube, orientation), rotations)

            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(zeros(dose_cube[:, :, self.slice].shape))

            self.dose_cube = None
            self.dose_contours = None

    def reset_segment_contours(self):
        """."""

        # 
        orientation, rotations, axis = self.orientations[
            self.parent.plane_cbox.currentText()]

        if (self.segment_masks is not None
                and self.segment_contours is not None):

            # 
            segment_masks = tuple(rot90(
                transpose(mask, orientation), rotations)
                for mask in self.segment_masks)

            # Loop over the segment contours
            for mask, contour in zip(segment_masks, self.segment_contours):

                # Update the segment contour lines
                contour.setData(zeros(mask[:, :, self.slice].shape))

            self.segment_masks = None
            self.segment_contours = None

    def reset_parent(self):
        """."""

        self.parent.plane_cbox.setCurrentText('axial')
        self.parent.slice_selection_pos.clear()
        self.parent.set_disabled((
            'plane_cbox', 'opacity_sbox', 'slice_selection_sbar'))

    def reset_images(self):
        """."""

        self.reset_ct()
        self.reset_dose()
        self.reset_dose_contours()
        self.reset_segment_contours()
        self.reset_parent()

    def change_orientation(self):
        """."""

        # 
        self.slice = self.parent.slice_selection_sbar.value()

        # 
        self.viewbox.enableAutoRange()

        # 
        self.update_images()

    def change_dose_opacity(self):
        """."""

        # 
        self.dose_image.setOpacity(self.parent.opacity_sbox.value()/100)

        # 
        self.update_dose()

    def change_image_slice(self):
        """."""

        # 
        self.slice = self.parent.slice_selection_sbar.value()

        # 
        self.update_images()