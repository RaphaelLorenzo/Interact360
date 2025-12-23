#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from tkinter import N
import numpy as np
import cv2
import json
import datetime
import time
import yaml
from typing import List, Dict, Optional, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from misc.print_utils import *
from misc.other_utils import *
from misc.visualize_utils import GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS, COCO_EDGES

# Import torch and RLE tools
import torch
from misc.rle_tools import decode_RLE, encode_RLE

MASK_RLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video decoding is handled by OpenCV VideoCapture (cv2.VideoCapture)
# which is already imported and works reliably with PyQt

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSlider, QPushButton, QCheckBox, QListWidget, QLabel, QSplitter,
        QGroupBox, QGridLayout, QListWidgetItem, QMessageBox, QStatusBar,
        QSpinBox, QTabWidget, QComboBox, QInputDialog, QAbstractItemView,
        QDialog, QDialogButtonBox, QRadioButton, QButtonGroup
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeySequence, QShortcut, QBrush
    PYQT_AVAILABLE = True
except ImportError:
    prError("PyQt6 not available. Please install with: pip install PyQt6")
    sys.exit(1)

# Import pyqtgraph for plotting
import pyqtgraph as pg
from pyqtgraph import PlotWidget as PGPlotWidget
pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('antialias', True)
PYQTGRAPH_AVAILABLE = True



class PlotWidget(QWidget):
    """Custom widget for displaying real-time plots using pyqtgraph"""
    
    def __init__(self, title="Plot", ylabel="Value", time_window_past=10, time_window_future=5, single_color=None, allow_values="nonzero", ylims=None):
        """
        allow_values: "nonzero", "strictlynonnegative", "nonnegative", "all"
        """
        super().__init__()
        self.title = title
        self.ylabel = ylabel
        self.time_window_past = time_window_past
        self.time_window_future = time_window_future
        self.single_color = single_color
        self.allow_values = allow_values
        self.ylims = ylims
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create the plot widget
        if PYQTGRAPH_AVAILABLE:
            self.plot_widget = PGPlotWidget(title=title)
            self.plot_widget.setLabel('left', ylabel)
            self.plot_widget.setLabel('bottom', 'Frame')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setBackground('w')  # White background
            
            # Store plot curves for different tracks
            self.plot_curves = {}  # track_id -> PlotCurveItem
            self.current_frame_line = None
            
            layout.addWidget(self.plot_widget)
            self.available = True
        else:
            # Create placeholder if pyqtgraph not available
            placeholder = QLabel(f"Plot: {title}\n(PyQtGraph not available)")
            placeholder.setStyleSheet("border: 1px solid gray; padding: 20px; color: gray; background-color: #f0f0f0;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)
            self.plot_widget = None
            self.available = False
        
        # Store plot data
        self.plot_data = {}  # track_id -> {'frames': [], 'values': []}
        self.current_frame = None
        
    def set_time_window(self, past_frames, future_frames):
        """Set the frame window for the plot"""
        self.time_window_past = past_frames
        self.time_window_future = future_frames
        
    def update_plot(self, current_frame, data_dict):
        """
        Update the plot with new data
        data_dict: {track_id: [(frame, value), (frame, value), ...]}
        """
        if not self.available:
            return
            
        self.current_frame = current_frame
        self.plot_data = {}  # track_id -> {'frames': [], 'values': []}, reinitialize the plot data every time

        # Add new data points
        for track_id, points in data_dict.items():
            if track_id not in self.plot_data:
                self.plot_data[track_id] = {'frames': [], 'values': []}
            
            self.plot_data[track_id]['frames'] = [v[0] for v in points]
            self.plot_data[track_id]['values'] = [v[1] for v in points]
        
        # Clean old data outside frame window
        frame_start = current_frame - self.time_window_past
        frame_end = current_frame + self.time_window_future
        
        for track_id in list(self.plot_data.keys()):
            frames = self.plot_data[track_id]['frames']
            values = self.plot_data[track_id]['values']
            
            # reorder by frames
            frames, values = zip(*sorted(zip(frames, values)))
            
            # Filter data within frame window
            valid_indices = [i for i, f in enumerate(frames) if frame_start <= f <= frame_end]
            
            if valid_indices:
                if self.allow_values == "nonzero":
                    self.plot_data[track_id]['frames'] = [frames[i] for i in valid_indices if values[i] != 0]
                    self.plot_data[track_id]['values'] = [values[i] for i in valid_indices if values[i] != 0]
                elif self.allow_values == "strictlynonnegative":
                    self.plot_data[track_id]['frames'] = [frames[i] for i in valid_indices if values[i] > 0]
                    self.plot_data[track_id]['values'] = [values[i] for i in valid_indices if values[i] > 0]
                elif self.allow_values == "nonnegative":
                    self.plot_data[track_id]['frames'] = [frames[i] for i in valid_indices if values[i] >= 0]
                    self.plot_data[track_id]['values'] = [values[i] for i in valid_indices if values[i] >= 0]
                elif self.allow_values == "all":
                    self.plot_data[track_id]['frames'] = [frames[i] for i in valid_indices]
                    self.plot_data[track_id]['values'] = [values[i] for i in valid_indices]
                else:
                    raise ValueError(f"Invalid allow_values: {self.allow_values}")
            else:
                # Remove track if no data in window
                del self.plot_data[track_id]
        
        self.redraw_plot()
    
    def redraw_plot(self):
        """Redraw the entire plot using pyqtgraph"""
        if not self.available or self.plot_widget is None:
            return
        
        # Clear existing curves
        for curve in self.plot_curves.values():
            self.plot_widget.removeItem(curve)
        self.plot_curves.clear()
        
        # Remove current frame line if it exists
        if self.current_frame_line is not None:
            self.plot_widget.removeItem(self.current_frame_line)
            self.current_frame_line = None
        
        # Colors for different tracks
        colors = [(98, 170, 174), (200, 150, 168), (213, 208, 44), (191, 14, 148), (150, 138, 149), 
                  (133, 161, 102), (105, 161, 43), (71, 120, 13), (13, 81, 212), (44, 150, 224), 
                  (243, 207, 71), (157, 248, 179), (121, 164, 71), (34, 93, 202), (175, 49, 4), 
                  (43, 248, 163), (38, 20, 240), (59, 191, 20), (142, 8, 82), (219, 88, 31), 
                  (37, 147, 24), (83, 137, 140), (218, 35, 175), (94, 237, 138), (35, 91, 70), 
                  (7, 55, 87), (223, 77, 210), (173, 146, 250), (1, 131, 67), (62, 12, 215)]
        
        if self.single_color:
            colors = [self.single_color]
    
        # Plot data for each track
        for i, (track_id, data) in enumerate(self.plot_data.items()):
            if data['frames'] and data['values']:
                color = colors[track_id % len(colors)]
                color = (color[2], color[1], color[0])  # Convert BGR to RGB
                                
                # Create curve for this track
                curve = self.plot_widget.plot(
                    data['frames'], 
                    data['values'], 
                    pen=pg.mkPen(color=color, width=2),
                    symbol='o', 
                    symbolSize=4,
                    symbolBrush=color,
                    name=f'Track {track_id}'
                )
                self.plot_curves[track_id] = curve
        
        # Add vertical line at current frame
        if self.current_frame is not None:
            self.current_frame_line = self.plot_widget.addLine(
                x=self.current_frame, 
                pen=pg.mkPen(color='red', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                label='Current Frame'
            )
        
        # Set frame window
        if self.current_frame is not None:
            frame_start = self.current_frame - self.time_window_past
            frame_end = self.current_frame + self.time_window_future
            self.plot_widget.setXRange(frame_start, frame_end, padding=0)
        
        if self.ylims:
            self.plot_widget.setYRange(self.ylims[0], self.ylims[1], padding=0.0)
    
    def clear_plot(self):
        """Clear all plot data"""
        if not self.available:
            return
            
        self.plot_data.clear()
        
        # Clear existing curves
        if hasattr(self, 'plot_curves'):
            for curve in self.plot_curves.values():
                self.plot_widget.removeItem(curve)
            self.plot_curves.clear()
        
        # Remove current frame line if it exists
        if hasattr(self, 'current_frame_line') and self.current_frame_line is not None:
            self.plot_widget.removeItem(self.current_frame_line)
            self.current_frame_line = None


class ImageDisplayWidget(QLabel):
    """Custom widget for displaying 360° images"""
    
    def __init__(self, visualizer=None):
        super().__init__()
        self.setMinimumSize(800, 400)
        self.setScaledContents(False)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_factor = (1.0, 1.0)
        self.image_size = (0, 0)
        self.original_pixmap = None
        self.visualizer = visualizer  # Reference to parent visualizer
        
    def set_image_with_scale(self, image_array):
        """Set image and store scale factor for coordinate conversion"""
        self.image_size = (image_array.shape[1], image_array.shape[0])
        
        # Convert numpy array to QImage
        if len(image_array.shape) == 3:
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            height, width = image_array.shape
            bytes_per_line = width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        self.original_pixmap = pixmap
        
        # Scale pixmap to fit widget while maintaining aspect ratio
        widget_size = self.size()
        if widget_size.width() > 0 and widget_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                widget_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        else:
            self.setPixmap(pixmap)
        
        self.update_scale_factors()
    
    def update_scale_factors(self):
        """Update scale factors based on current widget size and image size"""
        if self.image_size[0] == 0 or self.image_size[1] == 0:
            self.scale_factor = (1.0, 1.0)
            return
            
        widget_size = self.size()
        scale_x = self.image_size[0] / widget_size.width()
        scale_y = self.image_size[1] / widget_size.height()
        self.scale_factor = (scale_x, scale_y)
    
    def widget_to_image_coords(self, widget_x, widget_y):
        """Convert widget coordinates to image coordinates"""
        if self.image_size[0] == 0 or self.image_size[1] == 0:
            return None, None
        
        # Get the pixmap size (which is scaled to fit widget)
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None, None
        
        pixmap_size = pixmap.size()
        widget_size = self.size()
        
        # Calculate the actual displayed pixmap position (centered)
        pixmap_x_offset = (widget_size.width() - pixmap_size.width()) // 2
        pixmap_y_offset = (widget_size.height() - pixmap_size.height()) // 2
        
        # Convert widget coordinates to pixmap coordinates
        pixmap_x = widget_x - pixmap_x_offset
        pixmap_y = widget_y - pixmap_y_offset
        
        # Check if click is within pixmap bounds
        if pixmap_x < 0 or pixmap_x >= pixmap_size.width() or pixmap_y < 0 or pixmap_y >= pixmap_size.height():
            return None, None
        
        # Convert pixmap coordinates to image coordinates
        image_x = int(pixmap_x * self.image_size[0] / pixmap_size.width())
        image_y = int(pixmap_y * self.image_size[1] / pixmap_size.height())
        
        return image_x, image_y
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on the image"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.visualizer and self.visualizer.edit_mode:
                widget_x = event.position().x()
                widget_y = event.position().y()
                
                image_x, image_y = self.widget_to_image_coords(widget_x, widget_y)
                if image_x is not None and image_y is not None:
                    # Remap coordinates to account for horizontal roll
                    # np.roll shifts right for positive offset: Original[i] -> Rolled[(i + offset) % width]
                    # So to unroll: Rolled[x] -> Original[(x - offset) % width]
                    if self.visualizer.horizontal_roll_offset != 0:
                        image_width = self.image_size[0]
                        # Unroll: subtract the offset (with wrapping)
                        image_x = (image_x - self.visualizer.horizontal_roll_offset) % image_width
                    
                    # Check if Ctrl key is pressed for multi-selection
                    ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                    self.visualizer.select_track_by_click(image_x, image_y, ctrl_pressed)
        
        super().mousePressEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events to control frame slider"""
        if self.visualizer and self.visualizer.frame_slider:
            # Get the angle delta (positive for scrolling up/away, negative for scrolling down/toward)
            angle_delta = event.angleDelta().y()
            
            # Determine step size based on modifier keys
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Shift + wheel: jump by 10 frames
                step = 10
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl + wheel: jump by 5 frames
                step = 5
            else:
                # Normal wheel: move by 1 frame
                step = 1
            
            # Calculate new frame
            current_value = self.visualizer.frame_slider.value()
            if angle_delta < 0:
                # Scrolling down: go to previous frame(s)
                new_value = max(0, current_value - step)
            else:
                # Scrolling up: go to next frame(s)
                max_value = self.visualizer.frame_slider.maximum()
                new_value = min(max_value, current_value + step)
            
            # Update slider (which will trigger on_slider_change)
            self.visualizer.frame_slider.setValue(new_value)
        
        super().wheelEvent(event)
    
    def resizeEvent(self, event):
        """Handle widget resize to update scale factors and rescale image"""
        super().resizeEvent(event)
        
        if self.original_pixmap and not self.original_pixmap.isNull():
            widget_size = self.size()
            if widget_size.width() > 0 and widget_size.height() > 0:
                scaled_pixmap = self.original_pixmap.scaled(
                    widget_size, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.blockSignals(True)
                self.setPixmap(scaled_pixmap)
                self.blockSignals(False)
        
        self.update_scale_factors()


class EngagementCorrectionDialog(QDialog):
    """Dialog for correcting engagement levels of a track"""
    
    def __init__(self, parent, track_id, current_frame, num_frames, engagement_data):
        """
        Args:
            parent: Parent window
            track_id: Track ID to correct
            current_frame: Current frame index
            num_frames: Total number of frames
            engagement_data: Dict with 'frames' and 'engagement' lists for the track
        """
        super().__init__(parent)
        self.track_id = track_id
        self.current_frame = current_frame
        self.num_frames = num_frames
        self.engagement_data = engagement_data
        self.new_engagement = 0
        self.past_frames = 0
        self.future_frames = 15
        self.accepted_correction = False
        
        self.setWindowTitle(f"Engagement Correction - Track {track_id}")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Track info
        info_label = QLabel(f"Track ID: {track_id} | Current Frame: {current_frame}")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        # Engagement level selection
        engagement_group = QGroupBox("New Engagement Level")
        engagement_layout = QVBoxLayout(engagement_group)
        
        self.engagement_button_group = QButtonGroup(self)
        self.not_engaged_radio = QRadioButton("0 - Not Engaged")
        self.not_engaged_radio.setChecked(True)
        self.engaged_radio = QRadioButton("3 - Engaged")
        self.engagement_button_group.addButton(self.not_engaged_radio, 0)
        self.engagement_button_group.addButton(self.engaged_radio, 3)
        engagement_layout.addWidget(self.not_engaged_radio)
        engagement_layout.addWidget(self.engaged_radio)
        
        layout.addWidget(engagement_group)
        
        # Frame range selection
        frame_range_group = QGroupBox("Frame Range")
        frame_range_layout = QGridLayout(frame_range_group)
        
        frame_range_layout.addWidget(QLabel("Past frames:"), 0, 0)
        self.past_frames_spin = QSpinBox()
        self.past_frames_spin.setRange(0, 1000)
        self.past_frames_spin.setValue(0)
        self.past_frames_spin.valueChanged.connect(self.update_preview)
        frame_range_layout.addWidget(self.past_frames_spin, 0, 1)
        
        frame_range_layout.addWidget(QLabel("Future frames:"), 1, 0)
        self.future_frames_spin = QSpinBox()
        self.future_frames_spin.setRange(0, 1000)
        self.future_frames_spin.setValue(15)
        self.future_frames_spin.valueChanged.connect(self.update_preview)
        frame_range_layout.addWidget(self.future_frames_spin, 1, 1)
        
        layout.addWidget(frame_range_group)
        
        # Preview plot
        plot_group = QGroupBox("Engagement History Preview")
        plot_layout = QVBoxLayout(plot_group)
        
        if PYQTGRAPH_AVAILABLE:
            self.preview_plot = PGPlotWidget(title="Engagement Level Over Time")
            self.preview_plot.setLabel('left', 'Engagement Level')
            self.preview_plot.setLabel('bottom', 'Frame')
            self.preview_plot.showGrid(x=True, y=True, alpha=0.3)
            self.preview_plot.setBackground('w')
            self.preview_plot.setYRange(-0.5, 3.5, padding=0)
            plot_layout.addWidget(self.preview_plot)
        else:
            plot_placeholder = QLabel("Plot preview not available (PyQtGraph required)")
            plot_layout.addWidget(plot_placeholder)
            self.preview_plot = None
        
        layout.addWidget(plot_group)
        
        # Connect radio buttons
        self.not_engaged_radio.toggled.connect(self.update_preview)
        self.engaged_radio.toggled.connect(self.update_preview)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept_correction)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initial preview update
        self.update_preview()
    
    def update_preview(self):
        """Update the preview plot with current and corrected engagement levels"""
        if not PYQTGRAPH_AVAILABLE or self.preview_plot is None:
            return
        
        # Get selected engagement level
        if self.engaged_radio.isChecked():
            new_engagement = 3
        else:
            new_engagement = 0
        
        # Get frame range
        past_frames = self.past_frames_spin.value()
        future_frames = self.future_frames_spin.value()
        
        # Clear previous plot
        self.preview_plot.clear()
        
        # Get original engagement data
        orig_frames = self.engagement_data.get('frames', [])
        orig_engagement = self.engagement_data.get('engagement', [])
        
        # Calculate frame range for correction
        frame_start = max(0, self.current_frame - past_frames)
        frame_end = min(self.num_frames - 1, self.current_frame + future_frames)
        
        # Plot original engagement
        if orig_frames and orig_engagement:
            self.preview_plot.plot(
                orig_frames, orig_engagement,
                pen=pg.mkPen(color='blue', width=2),
                symbol='o', symbolSize=4,
                name='Original'
            )
        
        # Plot corrected engagement for the range
        corrected_frames = []
        corrected_engagement = []
        for frame in range(frame_start, frame_end + 1):
            corrected_frames.append(frame)
            corrected_engagement.append(new_engagement)
        
        if corrected_frames:
            self.preview_plot.plot(
                corrected_frames, corrected_engagement,
                pen=pg.mkPen(color='red', width=5, style=pg.QtCore.Qt.PenStyle.DashLine),
                symbol='x', symbolSize=12,
                name='Corrected'
            )
        
        # Add vertical line at current frame
        self.preview_plot.addLine(
            x=self.current_frame,
            pen=pg.mkPen(color='green', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label='Current Frame'
        )
        
        # Set X range to show relevant frames
        if orig_frames or corrected_frames:
            all_frames = list(set(orig_frames + corrected_frames))
            if all_frames:
                x_min = max(0, min(all_frames) - 10)
                x_max = min(self.num_frames - 1, max(all_frames) + 10)
                self.preview_plot.setXRange(x_min, x_max, padding=0.1)
    
    def accept_correction(self):
        """Accept the correction and store the values"""
        if self.engaged_radio.isChecked():
            self.new_engagement = 3
        else:
            self.new_engagement = 0
        
        self.past_frames = self.past_frames_spin.value()
        self.future_frames = self.future_frames_spin.value()
        self.accepted_correction = True
        self.accept()


class AllTracksEngagementDialog(QDialog):
    """Dialog for setting engagement level for all tracks in a frame range"""
    
    def __init__(self, parent, current_frame, num_frames):
        """
        Args:
            parent: Parent window
            current_frame: Current frame index
            num_frames: Total number of frames
        """
        super().__init__(parent)
        self.current_frame = current_frame
        self.num_frames = num_frames
        self.new_engagement = 0
        self.past_frames = 0
        self.future_frames = 0
        self.accepted = False
        
        self.setWindowTitle("Set Engagement for All Tracks")
        self.setMinimumSize(400, 200)
        
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(f"Current Frame: {current_frame} | Total Frames: {num_frames}")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(info_label)
        
        # Engagement level selection
        engagement_group = QGroupBox("New Engagement Level")
        engagement_layout = QVBoxLayout(engagement_group)
        
        self.engagement_button_group = QButtonGroup(self)
        self.not_engaged_radio = QRadioButton("0 - Not Engaged")
        self.not_engaged_radio.setChecked(True)
        self.engaged_radio = QRadioButton("3 - Engaged")
        self.engagement_button_group.addButton(self.not_engaged_radio, 0)
        self.engagement_button_group.addButton(self.engaged_radio, 3)
        engagement_layout.addWidget(self.not_engaged_radio)
        engagement_layout.addWidget(self.engaged_radio)
        
        layout.addWidget(engagement_group)
        
        # Frame range selection
        frame_range_group = QGroupBox("Frame Range")
        frame_range_layout = QGridLayout(frame_range_group)
        
        frame_range_layout.addWidget(QLabel("Past frames:"), 0, 0)
        self.past_frames_spin = QSpinBox()
        self.past_frames_spin.setRange(0, 10000)
        self.past_frames_spin.setValue(0)
        frame_range_layout.addWidget(self.past_frames_spin, 0, 1)
        
        frame_range_layout.addWidget(QLabel("Future frames:"), 1, 0)
        self.future_frames_spin = QSpinBox()
        self.future_frames_spin.setRange(0, 10000)
        self.future_frames_spin.setValue(0)
        frame_range_layout.addWidget(self.future_frames_spin, 1, 1)
        
        layout.addWidget(frame_range_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept_dialog)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def accept_dialog(self):
        """Accept the dialog and store the values"""
        if self.engaged_radio.isChecked():
            self.new_engagement = 3
        else:
            self.new_engagement = 0
        
        self.past_frames = self.past_frames_spin.value()
        self.future_frames = self.future_frames_spin.value()
        self.accepted = True
        self.accept()


class EpisodePreloadWorker(QThread):
    """Worker thread to preload next episode's data in background"""
    finished = pyqtSignal(int, dict, list, object, str, list)  # episode_idx, detections_dict, video_frames, video_cap, video_path, image_files
    error = pyqtSignal(int, str)  # episode_idx, error_message
    
    def __init__(self, visualizer, episode_idx):
        super().__init__()
        self.visualizer = visualizer
        self.episode_idx = episode_idx
        self._cancelled = False
    
    def cancel(self):
        """Cancel the preload operation"""
        self._cancelled = True
    
    def run(self):
        """Load episode data in background"""
        try:
            # Check for interruption request
            if self.isInterruptionRequested() or self._cancelled or self.episode_idx >= len(self.visualizer.episodes_list):
                return
            
            session, episode, associated_episodes_flags_dict = self.visualizer.episodes_list[self.episode_idx]
            episode_dir = self.visualizer.get_episode_dir(session, episode)
            annotations_dir = self.visualizer.get_annotations_dir(session, episode)
            
            # Check cancellation before proceeding
            if self.isInterruptionRequested() or self._cancelled:
                return
            
            # Determine image_files first
            image_files = []
            if self.visualizer.video_mode:
                video_path = self.visualizer.get_video_path(session, episode)
                if os.path.exists(video_path):
                    temp_cap = cv2.VideoCapture(video_path)
                    if temp_cap.isOpened():
                        frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        image_files = [f"frame_{i:06d}" for i in range(frame_count)]
                    temp_cap.release()
            else:
                if self.visualizer.use_mini:
                    images_dir = os.path.join(episode_dir, "miniatures_images_360_scale_4")
                else:
                    images_dir = os.path.join(episode_dir, "images_360")
                image_files = self.visualizer.get_image_files(images_dir)
            
            # Check cancellation
            if self.isInterruptionRequested() or self._cancelled:
                return
            
            # Load annotations
            detection_file = os.path.join(annotations_dir, "annotations_360.json")
            detections_dict = {}
            if os.path.exists(detection_file):
                detections_list = read_json_to_dic(detection_file)
                detections_list = self.visualizer.reorder_detections_list(detections_list)
                # Temporarily set image_files for conversion
                original_image_files = self.visualizer.image_files
                self.visualizer.image_files = image_files
                detections_dict = self.visualizer.convert_detections_list_to_dict(detections_list)
                # Restore original image_files
                self.visualizer.image_files = original_image_files
            
            # Check cancellation
            if self.isInterruptionRequested() or self._cancelled:
                return
            
            # Load video frames if video_mode and mini
            video_frames = []
            video_cap = None
            video_path = None
            
            if self.visualizer.video_mode and self.visualizer.use_mini:
                video_path = self.visualizer.get_video_path(session, episode)
                if os.path.exists(video_path):
                    try:
                        video_cap = cv2.VideoCapture(video_path)
                        if video_cap.isOpened():
                            frames_list = []
                            frame_idx = 0
                            while True:
                                if self.isInterruptionRequested() or self._cancelled:
                                    # Clean up and exit
                                    video_cap.release()
                                    return
                                
                                ret, frame = video_cap.read()
                                if not ret:
                                    break
                                frames_list.append(frame.copy())
                                frame_idx += 1
                            
                            video_cap.release()
                            video_cap = None
                            video_frames = frames_list
                    except Exception as e:
                        if not self._cancelled:
                            prWarning(f"Failed to preload video frames for episode {self.episode_idx}: {e}")
                        if video_cap is not None:
                            video_cap.release()
                        video_cap = None
            
            # Check cancellation before emitting signal
            if self.isInterruptionRequested() or self._cancelled:
                return
            
            # Emit finished signal with preloaded data
            self.finished.emit(self.episode_idx, detections_dict, video_frames, video_cap, video_path, image_files)
            
        except Exception as e:
            import traceback
            error_msg = f"Preload error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(self.episode_idx, error_msg)


class Interact360Visualizer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Report-based filtering
        self.report_data = None
        self.report_tracks_by_episode = {}  # (recording, episode) -> list of track dicts with status
        self.report_track_status = {}  # unique_track_id -> "tp"/"fp"/"tn"/"fn"
        
        # Load report if provided
        if args.select_tracks_from_report:
            self.load_report(args.select_tracks_from_report, args.select_tracks_based_on_prediction, args.prediction_threshold)
        
        # Find episodes (will be filtered if report is loaded)
        self.episodes_list = self.find_episodes()
        self.current_episode_idx = 0
        
        # Data storage
        self.image_files = []
        self.detections = {}
        self.current_frame = 0
        
        # Track data for plots
        self.track_engagement_data = {}  # track_id -> {'frames': [], 'engagement': []}
        self.track_mask_size_data = {}  # track_id -> {'frames': [], 'mask_size': []}
        self.track_intersection_pixels_data = {}  # track_id -> {'frames': [], 'intersection_pixels': []}
        
        # Interaction mask
        self.interaction_mask = None
        self.interact_zone_tracking = None  # Dynamic interaction zones per frame
        self.load_interaction_mask()
        
        # Display options
        self.show_masks = False
        self.show_boxes = True
        self.show_track_ids = True
        self.show_interaction_highlights = True
        self.show_interaction_zone = True  # Option to constantly display interaction zone
        self.mask_opacity = 70  # Default 70% opacity
        
        # Pose visualization options
        self.show_vitpose_poses = True
        self.show_sapiens_poses = True
        self.vitpose_min_score = 0.6  # Minimum score for ViTPose keypoints
        self.sapiens_min_score = 0.6  # Minimum score for Sapiens keypoints
        
        # Track validity options
        self.show_invalid_tracks = True  # Show tracks that are not valid
        
        # Miniature image support
        self.use_mini = args.miniatures
        prInfo(f"Using miniature images: {self.use_mini}")
        self.scale_factor = 4.0 if self.use_mini else 1.0  # Miniature images are 4x smaller
        self.full_image_height = None  # Full size image height (for mask decoding) - set when first image is loaded
        self.full_image_width = None  # Full size image width (for mask decoding) - set when first image is loaded
        
        # Video mode support
        self.video_mode = args.video_mode
        self.video_cap = None  # OpenCV VideoCapture instance for current episode
        self.video_frames = None  # Cached video frames (numpy array)
        self.video_path = None  # Path to current video file
        if self.video_mode:
            prInfo("Using video mode - loading frames from video files using OpenCV")
        
        # Preloading support
        self.preload_worker = None  # Background worker for preloading
        self.preloaded_data = {}  # Cache for preloaded episode data: {episode_idx: {'detections': ..., 'video_frames': ..., 'video_path': ...}}
        
        # Image rolling to center interaction mask
        self.enable_image_rolling = True  # Enable/disable image rolling (enabled by default)
        self.horizontal_roll_offset = 0  # Horizontal pixel offset for rolling the image
        
        # Edit mode options
        self.edit_mode = True  # Enable edit mode by default
        self.edited_detections = {}  # Store local modifications
        self.next_available_id = 0  # Next available track ID for splitting
        self.current_episodes_flags_path = None  # Path to episodes_flag.json for current episode
        
        # Plot time windows
        self.plot_window_past = 50
        self.plot_window_future = 20
        
        # Track colors
        self.colors = [
            (98, 170, 174), (200, 150, 168), (213, 208, 44), (191, 14, 148), (150, 138, 149), 
            (133, 161, 102), (105, 161, 43), (71, 120, 13), (13, 81, 212), (44, 150, 224), 
            (243, 207, 71), (157, 248, 179), (121, 164, 71), (34, 93, 202), (175, 49, 4), 
            (43, 248, 163), (38, 20, 240), (59, 191, 20), (142, 8, 82), (219, 88, 31), 
            (37, 147, 24), (83, 137, 140), (218, 35, 175), (94, 237, 138), (35, 91, 70), 
            (7, 55, 87), (223, 77, 210), (173, 146, 250), (1, 131, 67), (62, 12, 215)
        ]
        
        # Selected track for highlighting
        self.selected_track = None
        
        # Play/Pause functionality
        self.is_playing = False
        self.play_skip = 1
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.auto_speed_adjustment = True  # Enable automatic speed adjustment by default
        
        self.init_ui()
        self.load_episode_data()
        
        # Initialize edit mode since it's enabled by default
        if self.edit_mode:
            self.initialize_edited_detections()
            self.calculate_next_available_id()
        
    def create_episodes_flags_file(self, episodes_flags_path, episodes_desc_path):
        episodes_desc_dict = read_json_to_dic(episodes_desc_path) # list like : [{"session": "2022_09_21_astor_place_recycle", "episode": "0000", "start_time": 1756981182.0, "end_time": 1756981301.9999995}...]
        episodes_flags_dict = {} # dict like {"0000": {"start_time": 1751876686, "end_time": 1751876704, "start_time_formatted": "2025-07-07 10:24:46", "end_time_formatted": "2025-07-07 10:25:04", "flag": "valid,checked"}...}
        for episode_desc in episodes_desc_dict:
            episodes_flags_dict[episode_desc["episode"]] = {"start_time": episode_desc["start_time"], "end_time": episode_desc["end_time"], "start_time_formatted": datetime.datetime.fromtimestamp(episode_desc["start_time"]).strftime("%Y-%m-%d %H:%M:%S"), "end_time_formatted": datetime.datetime.fromtimestamp(episode_desc["end_time"]).strftime("%Y-%m-%d %H:%M:%S"), "flag": "valid"}
            
        write_dic_to_json_file(episodes_flags_dict, episodes_flags_path)
        prInfo(f"Created episodes flags file: {episodes_flags_path}")
    
    def get_episode_dir(self, session, episode):
        """Get the episode directory path (for images and interaction zone tracking)"""
        if self.args.splitted_dataset:
            return os.path.join(self.args.directory, "dataset", session, "episodes", episode)
        else:
            return os.path.join(self.args.directory, session, "episodes", episode)
    
    def get_annotations_dir(self, session, episode):
        """Get the annotations directory path (for annotations_360.json)"""
        if self.args.splitted_dataset:
            return os.path.join(self.args.directory, "annotations", session, "episodes", episode)
        else:
            return os.path.join(self.args.directory, session, "episodes", episode)
    
    def get_video_path(self, session, episode):
        """Get the video file path based on video_mode, miniatures, and splitted_dataset settings"""
        if self.use_mini:
            video_filename = "images_360_mini.mp4"
            if self.args.splitted_dataset:
                video_dir = os.path.join(self.args.directory, "video_mini", session, "episodes", episode)
            else:
                video_dir = self.get_episode_dir(session, episode)
        else:
            video_filename = "images_360_full.mp4"
            if self.args.splitted_dataset:
                video_dir = os.path.join(self.args.directory, "video_full", session, "episodes", episode)
            else:
                video_dir = self.get_episode_dir(session, episode)
        
        video_path = os.path.join(video_dir, video_filename)
        return video_path
    
    def load_report(self, report_path, prediction_filter, custom_threshold=None):
        """Load report YAML file and compute track statuses (TP/FP/TN/FN)
        
        Args:
            report_path: Path to the report YAML file
            prediction_filter: Filter to apply ("tp", "fp", "tn", "fn", or "all")
            custom_threshold: Optional custom threshold for predicted_score. If provided,
                            predicted_label is computed as: predicted_label = 1 if predicted_score >= threshold, else 0
        """
        if not os.path.exists(report_path):
            prError(f"Report file does not exist: {report_path}")
            return
        
        try:
            with open(report_path, 'r') as f:
                self.report_data = yaml.safe_load(f)
            
            prInfo(f"Loaded report from {report_path}")
            
            # Use custom threshold if provided
            if custom_threshold is not None:
                prInfo(f"Using custom threshold {custom_threshold} to determine predicted_label from predicted_score")
            else:
                prInfo("Using predicted_label from report")
            
            # Compute TP/FP/TN/FN for each track
            self.report_track_status = {}
            self.report_tracks_by_episode = {}
            
            for item in self.report_data.get('item_results', []):
                unique_track_id = item.get('unique_track_id', '')
                predicted_score = item.get('predicted_score', 0.0)
                ground_truth_label = item.get('ground_truth_label', 0)
                recording = item.get('recording', '')
                episode = item.get('episode', '')
                frame_indexes = item.get('frame_indexes', [])
                
                # Determine predicted_label: use custom threshold if provided, otherwise use from report
                if custom_threshold is not None:
                    predicted_label = 1 if predicted_score >= custom_threshold else 0
                else:
                    predicted_label = item.get('predicted_label', 0)
                
                # Determine status
                if predicted_label == 1 and ground_truth_label == 1:
                    status = "tp"
                elif predicted_label == 1 and ground_truth_label == 0:
                    status = "fp"
                elif predicted_label == 0 and ground_truth_label == 0:
                    status = "tn"
                elif predicted_label == 0 and ground_truth_label == 1:
                    status = "fn"
                else:
                    status = "unknown"
                
                self.report_track_status[unique_track_id] = status
                
                # Group by episode
                episode_key = (recording, episode)
                if episode_key not in self.report_tracks_by_episode:
                    self.report_tracks_by_episode[episode_key] = []
                
                track_info = {
                    'unique_track_id': unique_track_id,
                    'status': status,
                    'predicted_label': predicted_label,
                    'ground_truth_label': ground_truth_label,
                    'predicted_score': item.get('predicted_score', 0.0),
                    'first_frame': min(frame_indexes) if frame_indexes else None,
                    'last_frame': max(frame_indexes) if frame_indexes else None,
                    'frame_indexes': frame_indexes
                }
                self.report_tracks_by_episode[episode_key].append(track_info)
            
            # Filter tracks based on prediction_filter
            if prediction_filter != "all":
                filtered_episodes = {}
                for episode_key, tracks in self.report_tracks_by_episode.items():
                    filtered_tracks = [t for t in tracks if t['status'] == prediction_filter]
                    if filtered_tracks:
                        filtered_episodes[episode_key] = filtered_tracks
                self.report_tracks_by_episode = filtered_episodes
                prInfo(f"Filtered to {len(filtered_episodes)} episodes with {prediction_filter} tracks")
            else:
                prInfo(f"Loaded {len(self.report_tracks_by_episode)} episodes from report")
                
        except Exception as e:
            prError(f"Error loading report: {e}")
            import traceback
            traceback.print_exc()
    
    def find_episodes(self):
        """Find available episodes in the output directory"""
        episodes_list = []
        
        if not os.path.exists(self.args.directory):
            prError(f"Output directory does not exist: {self.args.directory}")
            return episodes_list
        
        
        # If we are in a "splitted_dataset" mode, then the data and the annotations are separated into different directories : 
        # DIRECTORY/dataset/recording_name/ should contain the episodes.json and episodes_flag.json files
        # DIRECTORY/dataset/recording_name/episodes/XXX/images_360 should contain the images (if they have been extracted to be used by default) and eventually miniatures_images_360_scale_4 (if miniatures are used)
        # DIRECTORY/annotations/recording_name/episodes/XXX/annotations_360.json should contain the tracking annotations (and the filenames of images_360)
        # DIRECTORY/video_mini/recording_name/episodes/XXX/images_360_mini.mp4 should contain the mini video (to be used if video mode and miniatures are used)
        # DIRECTORY/video_full/recording_name/episodes/XXX/images_360_full.mp4 should contain the full video (to be used if video mode)
        
        # Determine which recording names to process
        recording_names_to_process = []
        
        if self.report_data is not None:
            # Extract all unique recording names from the report
            recording_names_to_process = list(set([recording for recording, _ in self.report_tracks_by_episode.keys()]))
            prInfo(f"Found {len(recording_names_to_process)} recording(s) in report: {recording_names_to_process}")
        else:
            # Use the specified recording name (has default value from argparse)
            if self.args.recording_name:
                recording_names_to_process = [self.args.recording_name]
            else:
                prError("No recording name specified and no report loaded. Please specify --recording_name or use --select_tracks_from_report")
                return episodes_list
        
        # Process each recording name
        for recording_name in recording_names_to_process:
            if self.args.splitted_dataset:
                dataset_dir = os.path.join(self.args.directory, "dataset", recording_name)
                annotations_dir = os.path.join(self.args.directory, "annotations", recording_name)
            else:
                dataset_dir = os.path.join(self.args.directory, recording_name)
                annotations_dir = os.path.join(self.args.directory, recording_name)

            if not os.path.exists(dataset_dir):
                prWarning(f"Recording name does not exist: {recording_name} in directory {self.args.directory} (looking for path {dataset_dir})")
                continue
            
            episodes_dir = os.path.join(dataset_dir, "episodes")
            episodes_desc_path = os.path.join(dataset_dir, "episodes.json") # this one is mandatory
            if not os.path.exists(episodes_desc_path):
                prWarning(f"Episodes description file does not exist: {episodes_desc_path}")
                continue
                
            episodes_flags_path = os.path.join(dataset_dir, "episodes_flag.json")
            if os.path.exists(episodes_flags_path):
                episodes_flags_dict = read_json_to_dic(episodes_flags_path)
                prInfo(f"Got episodes flags for {recording_name}: {len(episodes_flags_dict)} episodes")
            else:
                # no file found create if from episodes description
                self.create_episodes_flags_file(episodes_flags_path, episodes_desc_path)
                prSuccess(f"Created episodes flags file: {episodes_flags_path}")
            
            associated_episodes_flags_dict = read_json_to_dic(episodes_flags_path)
            
            if os.path.exists(episodes_dir):
                # Find episode subdirectories (like 0000)
                for ep in os.listdir(episodes_dir):
                    ep_path = os.path.join(episodes_dir, ep)
                    if os.path.isdir(ep_path) and ep.isdigit():
                        # If report filtering is enabled, only include episodes that have matching tracks
                        if self.report_data is not None:
                            episode_key = (recording_name, ep)
                            if episode_key in self.report_tracks_by_episode:
                                episodes_list.append((recording_name, ep, associated_episodes_flags_dict))
                        else:
                            episodes_list.append((recording_name, ep, associated_episodes_flags_dict))
                    
        episodes_list.sort()
        if self.report_data is not None:
            prInfo(f"Found {len(episodes_list)} episodes across {len(recording_names_to_process)} recording(s) (filtered by report)")
        else:
            prInfo(f"Found {len(episodes_list)} episodes")
        return episodes_list
    
    def load_interaction_mask(self):
        """Load the default interaction mask from assets"""
        interaction_mask_path = os.path.join(os.path.dirname(__file__), "assets", "default_mask_intersection_zone.png")
        
        if os.path.exists(interaction_mask_path):
            # Load mask as grayscale
            mask_image = cv2.imread(interaction_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is not None:
                # Convert to boolean mask (assuming white areas are interaction zones)
                self.interaction_mask = mask_image > 128
                prInfo(f"Loaded interaction mask: {interaction_mask_path} ({mask_image.shape})")
            else:
                prWarning(f"Failed to load interaction mask image: {interaction_mask_path}")
                self.interaction_mask = None
        else:
            prWarning(f"Interaction mask not found: {interaction_mask_path}")
            self.interaction_mask = None
    
    def load_interact_zone_tracking(self, episode_dir):
        """Load dynamic interaction zone tracking if available"""
        interact_zone_tracking_path = os.path.join(episode_dir, "interaction_zone_tracking_360.json")
        
        if os.path.exists(interact_zone_tracking_path):
            self.interact_zone_tracking = read_json_to_dic(interact_zone_tracking_path)
            prInfo(f"Loaded interact zone tracking: {interact_zone_tracking_path} with {len(self.interact_zone_tracking)} frames")
            return True

        else:
            prInfo(f"No interact zone tracking file found: {interact_zone_tracking_path}")
            self.interact_zone_tracking = None
            return False
    
    def get_track_color(self, track_id):
        """Get the color for a specific track ID"""
        color_idx = track_id % len(self.colors)
        return self.colors[color_idx]
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Interact360 Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_layout = QHBoxLayout(central_widget)
        central_layout.addWidget(main_splitter)
        
        # Left panel: Controls
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(450)
        control_panel.setMinimumWidth(250)
        main_splitter.addWidget(control_panel)
        
        # Center panel: Image display
        image_section = self.create_image_section()
        main_splitter.addWidget(image_section)
        
        # Right panel: Plots
        plots_section = self.create_plots_section()
        plots_section.setMaximumWidth(400)
        plots_section.setMinimumWidth(300)
        main_splitter.addWidget(plots_section)
        
        # Set splitter proportions
        main_splitter.setSizes([450, 600, 350])
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
    
    def create_control_panel(self):
        """Create the control panel with all options"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Episode selection
        episode_group = QGroupBox("Episode Selection")
        episode_layout = QVBoxLayout(episode_group)
        
        # Episode dropdown
        self.episode_combo = QComboBox()
        for i, (session, ep, associated_episodes_flags_dict) in enumerate(self.episodes_list):
            episode_flag = associated_episodes_flags_dict[ep]["flag"]
            item_text = f"{session} | Ep. {ep} | {episode_flag}"
            self.episode_combo.addItem(item_text)
        self.episode_combo.currentIndexChanged.connect(self.change_episode)
        episode_layout.addWidget(self.episode_combo)
        
        # Episode navigation
        nav_layout = QHBoxLayout()
        self.prev_episode_btn = QPushButton("◀ Previous [Ctrl+B]")
        self.prev_episode_btn.clicked.connect(self.previous_episode)
        nav_layout.addWidget(self.prev_episode_btn)
        
        self.next_episode_btn = QPushButton("Next ▶ [Ctrl+N]")
        self.next_episode_btn.clicked.connect(self.next_episode)
        nav_layout.addWidget(self.next_episode_btn)
        episode_layout.addLayout(nav_layout)
        
        layout.addWidget(episode_group)
        
        # Play/Pause controls
        play_group = QGroupBox("Playback")
        play_layout = QHBoxLayout(play_group)
        
        self.play_pause_btn = QPushButton("▶ Play [Space]")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        play_layout.addWidget(self.play_pause_btn)
        
        # Skip parameter
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Skip:"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 30)
        self.skip_spin.setValue(self.play_skip)
        self.skip_spin.valueChanged.connect(self.update_skip)
        skip_layout.addWidget(self.skip_spin)
        play_layout.addLayout(skip_layout)
        
        # Auto speed adjustment
        self.auto_speed_cb = QCheckBox("AutoSpeed")
        self.auto_speed_cb.setChecked(True)
        self.auto_speed_cb.toggled.connect(self.toggle_auto_speed_adjustment)
        play_layout.addWidget(self.auto_speed_cb)
        
        layout.addWidget(play_group)
        
        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        
        # Group "Show" checkboxes 3 per row, removing "Show" prefix
        self.show_masks_cb = QCheckBox("Masks [M]")
        self.show_masks_cb.setChecked(False)
        self.show_masks_cb.toggled.connect(self.toggle_show_masks)
        options_layout.addWidget(self.show_masks_cb, 0, 0)
        
        self.show_boxes_cb = QCheckBox("Bounding Boxes [B]")
        self.show_boxes_cb.setChecked(True)
        self.show_boxes_cb.toggled.connect(self.toggle_show_boxes)
        options_layout.addWidget(self.show_boxes_cb, 0, 1)
        
        self.show_track_ids_cb = QCheckBox("Track IDs [T]")
        self.show_track_ids_cb.setChecked(True)
        self.show_track_ids_cb.toggled.connect(self.toggle_show_track_ids)
        options_layout.addWidget(self.show_track_ids_cb, 0, 2)
        
        self.show_interaction_highlights_cb = QCheckBox("Int. Highlights [I]")
        self.show_interaction_highlights_cb.setChecked(True)
        self.show_interaction_highlights_cb.setEnabled(self.interaction_mask is not None)
        self.show_interaction_highlights_cb.toggled.connect(self.toggle_show_interaction_highlights)
        options_layout.addWidget(self.show_interaction_highlights_cb, 1, 0)
        
        self.show_interaction_zone_cb = QCheckBox("Int. Zone [Z]")
        self.show_interaction_zone_cb.setChecked(True)
        self.show_interaction_zone_cb.setEnabled(self.interaction_mask is not None)
        self.show_interaction_zone_cb.toggled.connect(self.toggle_show_interaction_zone)
        options_layout.addWidget(self.show_interaction_zone_cb, 1, 1)
        
        # Pose visualization options
        self.show_vitpose_poses_cb = QCheckBox("ViTPose [P]")
        self.show_vitpose_poses_cb.setChecked(True)
        self.show_vitpose_poses_cb.toggled.connect(self.toggle_show_vitpose_poses)
        options_layout.addWidget(self.show_vitpose_poses_cb, 1, 2)
        
        self.show_sapiens_poses_cb = QCheckBox("Sapiens [Shift+P]")
        self.show_sapiens_poses_cb.setChecked(True)
        self.show_sapiens_poses_cb.toggled.connect(self.toggle_show_sapiens_poses)
        options_layout.addWidget(self.show_sapiens_poses_cb, 2, 0)
        
        self.show_invalid_tracks_cb = QCheckBox("Invalid Tracks [V]")
        self.show_invalid_tracks_cb.setChecked(True)
        self.show_invalid_tracks_cb.toggled.connect(self.toggle_show_invalid_tracks)
        options_layout.addWidget(self.show_invalid_tracks_cb, 2, 1)
        
        self.enable_image_rolling_cb = QCheckBox("Image Rolling")
        self.enable_image_rolling_cb.setChecked(True)
        self.enable_image_rolling_cb.toggled.connect(self.toggle_enable_image_rolling)
        options_layout.addWidget(self.enable_image_rolling_cb, 2, 2)
        
        # ViTPose minimum score control
        # vitpose_score_layout = QHBoxLayout()
        # vitpose_score_layout.addWidget(QLabel("ViTPose Min Score:"))
        # self.vitpose_score_spin = QSpinBox()
        # self.vitpose_score_spin.setRange(0, 100)
        # self.vitpose_score_spin.setValue(int(self.vitpose_min_score * 100))
        # self.vitpose_score_spin.setSuffix("%")
        # self.vitpose_score_spin.valueChanged.connect(self.update_vitpose_min_score)
        # vitpose_score_layout.addWidget(self.vitpose_score_spin)
        # options_layout.addLayout(vitpose_score_layout, 3, 0, 1, 3)
        
        # Sapiens minimum score control
        # sapiens_score_layout = QHBoxLayout()
        # sapiens_score_layout.addWidget(QLabel("Sapiens Min Score:"))
        # self.sapiens_score_spin = QSpinBox()
        # self.sapiens_score_spin.setRange(0, 100)
        # self.sapiens_score_spin.setValue(int(self.sapiens_min_score * 100))
        # self.sapiens_score_spin.setSuffix("%")
        # self.sapiens_score_spin.valueChanged.connect(self.update_sapiens_min_score)
        # sapiens_score_layout.addWidget(self.sapiens_score_spin)
        # options_layout.addLayout(sapiens_score_layout, 4, 0, 1, 3)
        
        # Mask opacity control
        mask_opacity_layout = QHBoxLayout()
        mask_opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.mask_opacity_spin = QSpinBox()
        self.mask_opacity_spin.setRange(0, 100)
        self.mask_opacity_spin.setValue(self.mask_opacity)
        self.mask_opacity_spin.setSuffix("%")
        self.mask_opacity_spin.valueChanged.connect(self.update_mask_opacity)
        mask_opacity_layout.addWidget(self.mask_opacity_spin)
        options_layout.addLayout(mask_opacity_layout, 5, 0, 1, 3)
        
        layout.addWidget(options_group)
        
        # Edit mode controls
        edit_group = QGroupBox("Edit Mode")
        edit_layout = QVBoxLayout(edit_group)
        
        # Edit mode toggle
        self.edit_mode_cb = QCheckBox("Edit Mode [Ctrl+E]")
        self.edit_mode_cb.setChecked(True)  # Enable edit mode by default
        self.edit_mode_cb.toggled.connect(self.toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_cb)
        
        # Track editing controls (visible by default since edit mode is enabled)
        self.track_edit_widget = QWidget()
        track_edit_layout = QVBoxLayout(self.track_edit_widget)
        
        # Validity editing
        validity_group = QGroupBox("Edit Track Validity")
        validity_layout = QVBoxLayout(validity_group)
        
        validity_buttons_layout = QHBoxLayout()
        self.set_valid_btn = QPushButton("Set Valid [V]")
        self.set_valid_btn.clicked.connect(self.set_track_validity_valid)
        validity_buttons_layout.addWidget(self.set_valid_btn)
        
        self.set_invalid_btn = QPushButton("Set Invalid [Ctrl+F]")
        self.set_invalid_btn.clicked.connect(self.set_track_validity_invalid)
        validity_buttons_layout.addWidget(self.set_invalid_btn)
        
        self.set_custom_btn = QPushButton("Set Custom [X]")
        self.set_custom_btn.clicked.connect(self.set_track_validity_custom)
        validity_buttons_layout.addWidget(self.set_custom_btn)
        
        validity_layout.addLayout(validity_buttons_layout)
        track_edit_layout.addWidget(validity_group)
        
        # Track operations
        operations_group = QGroupBox("Track Operations")
        operations_layout = QVBoxLayout(operations_group)
        
        self.split_track_btn = QPushButton("Split Track [S]")
        self.split_track_btn.clicked.connect(self.split_track)
        operations_layout.addWidget(self.split_track_btn)
        
        self.merge_tracks_btn = QPushButton("Merge Selected Tracks")
        self.merge_tracks_btn.clicked.connect(self.merge_tracks)
        operations_layout.addWidget(self.merge_tracks_btn)
        
        track_edit_layout.addWidget(operations_group)
        
        # Engagement correction
        engagement_group = QGroupBox("Engagement Correction")
        engagement_layout = QVBoxLayout(engagement_group)
        
        # First row: Correction buttons side by side
        correction_row = QHBoxLayout()
        self.engagement_correction_btn = QPushButton("Correction [E]")
        self.engagement_correction_btn.clicked.connect(self.open_engagement_correction_dialog)
        correction_row.addWidget(self.engagement_correction_btn)

        self.set_all_tracks_engagement_btn = QPushButton("All Tracks Correction")
        self.set_all_tracks_engagement_btn.clicked.connect(self.open_all_tracks_engagement_dialog)
        correction_row.addWidget(self.set_all_tracks_engagement_btn)
        engagement_layout.addLayout(correction_row)
        
        # Second row: Quick engagement buttons side by side
        quick_engagement_row = QHBoxLayout()
        self.set_engagement_zero_btn = QPushButton("To 0 (±15 frames) [Ctrl+X]")
        self.set_engagement_zero_btn.clicked.connect(self.set_engagement_to_zero)
        quick_engagement_row.addWidget(self.set_engagement_zero_btn)
        
        self.set_engagement_three_btn = QPushButton("To 3 (+15 frames) [Ctrl+W]")
        self.set_engagement_three_btn.clicked.connect(self.set_engagement_to_three)
        quick_engagement_row.addWidget(self.set_engagement_three_btn)
        engagement_layout.addLayout(quick_engagement_row)
        
        track_edit_layout.addWidget(engagement_group)
        
        # Episode flag updates
        flags_group = QGroupBox("Update Episode Flag")
        flags_layout = QVBoxLayout(flags_group)
        
        flags_buttons_layout = QHBoxLayout()
        self.append_checked_btn = QPushButton("Append ,checked [C]")
        self.append_checked_btn.clicked.connect(self.append_checked_flag)
        flags_buttons_layout.addWidget(self.append_checked_btn)
        
        self.set_invalid_flag_btn = QPushButton("Set Invalid")
        self.set_invalid_flag_btn.clicked.connect(self.set_invalid_flag)
        flags_buttons_layout.addWidget(self.set_invalid_flag_btn)
        
        self.set_valid_flag_btn = QPushButton("Set Valid")
        self.set_valid_flag_btn.clicked.connect(self.set_valid_flag)
        flags_buttons_layout.addWidget(self.set_valid_flag_btn)
        
        flags_layout.addLayout(flags_buttons_layout)
        track_edit_layout.addWidget(flags_group)
        
        # Save changes
        save_layout = QHBoxLayout()
        self.save_changes_btn = QPushButton("Save Changes [Ctrl+S]")
        self.save_changes_btn.clicked.connect(self.save_changes)
        save_layout.addWidget(self.save_changes_btn)
        
        self.discard_changes_btn = QPushButton("Discard Changes [Ctrl+D]")
        self.discard_changes_btn.clicked.connect(self.discard_changes)
        save_layout.addWidget(self.discard_changes_btn)
        
        track_edit_layout.addLayout(save_layout)
        
        edit_layout.addWidget(self.track_edit_widget)
        self.track_edit_widget.setVisible(True)  # Visible by default since edit mode is enabled
        
        layout.addWidget(edit_group)
        
        # Track list
        track_group = QGroupBox("Tracks")
        track_layout = QVBoxLayout(track_group)
        
        self.track_list = QListWidget()
        self.track_list.setMaximumHeight(200)
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)  # Allow multiple selection
        self.track_list.itemClicked.connect(self.on_track_selected)
        track_layout.addWidget(self.track_list)
        
        # Clear selection button
        self.clear_selection_btn = QPushButton("Clear Selection [Esc]")
        self.clear_selection_btn.clicked.connect(self.clear_track_selection)
        track_layout.addWidget(self.clear_selection_btn)
        
        layout.addWidget(track_group)
        
        # Report tracks section (only shown if report is loaded)
        self.report_tracks_group = None
        self.report_tracks_list = None
        if self.report_data is not None:
            report_tracks_group = QGroupBox("Report Tracks")
            report_tracks_layout = QVBoxLayout(report_tracks_group)
            
            self.report_tracks_list = QListWidget()
            self.report_tracks_list.setMaximumHeight(200)
            self.report_tracks_list.itemDoubleClicked.connect(self.on_report_track_double_clicked)
            report_tracks_layout.addWidget(self.report_tracks_list)
            
            self.report_tracks_group = report_tracks_group
            layout.addWidget(report_tracks_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_image_section(self):
        """Create the image display section"""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Title
        title_label = QLabel("360° Camera View")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)
        
        # Image display
        self.image_widget = ImageDisplayWidget(self)
        layout.addWidget(self.image_widget, 4)
        
        # Info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setMaximumHeight(30)
        layout.addWidget(self.info_label)
        
        # Navigation controls
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        
        self.prev_btn = QPushButton("◀ Previous [←]")
        self.prev_btn.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_btn)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_change)
        nav_layout.addWidget(self.frame_slider)
        
        self.next_btn = QPushButton("Next ▶ [→]")
        self.next_btn.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_btn)
        
        self.frame_label = QLabel("0/0")
        nav_layout.addWidget(self.frame_label)
        
        nav_widget.setMaximumHeight(50)
        layout.addWidget(nav_widget)
        
        return section
    
    def create_plots_section(self):
        """Create the plots section"""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Title
        title_label = QLabel("Track Analytics")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)
        
        # Create plot widgets
        self.intersection_pixels_plot = PlotWidget(
            "Intersection Pixels", "Pixels", 
            self.plot_window_past, self.plot_window_future, 
            allow_values="nonnegative",
            ylims=[0, 5000]
        )
        layout.addWidget(self.intersection_pixels_plot)
        
        self.engagement_plot = PlotWidget(
            "Track Engagement", "Engagement Level", 
            self.plot_window_past, self.plot_window_future, 
            allow_values="nonnegative", ylims=[0, 3.5]
        )
        layout.addWidget(self.engagement_plot)
        
        self.mask_size_plot = PlotWidget(
            "Mask Size", "Pixels", 
            self.plot_window_past, self.plot_window_future, 
            allow_values="strictlynonnegative",
            ylims = [0, 300000]
        )
        layout.addWidget(self.mask_size_plot)
        
        return section
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Navigation shortcuts
        QShortcut(QKeySequence("Left"), self, self.prev_frame)
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        QShortcut(QKeySequence("Ctrl+Left"), self, self.prev_frame_10)
        QShortcut(QKeySequence("Ctrl+Right"), self, self.next_frame_10)
        
        # Display shortcuts
        QShortcut(QKeySequence("M"), self, lambda: self.show_masks_cb.setChecked(not self.show_masks_cb.isChecked()))
        QShortcut(QKeySequence("B"), self, lambda: self.show_boxes_cb.setChecked(not self.show_boxes_cb.isChecked()))
        QShortcut(QKeySequence("T"), self, lambda: self.show_track_ids_cb.setChecked(not self.show_track_ids_cb.isChecked()))
        QShortcut(QKeySequence("I"), self, lambda: self.show_interaction_highlights_cb.setChecked(not self.show_interaction_highlights_cb.isChecked()))
        QShortcut(QKeySequence("Z"), self, lambda: self.show_interaction_zone_cb.setChecked(not self.show_interaction_zone_cb.isChecked()))
        QShortcut(QKeySequence("P"), self, lambda: self.show_vitpose_poses_cb.setChecked(not self.show_vitpose_poses_cb.isChecked()))
        QShortcut(QKeySequence("Shift+P"), self, lambda: self.show_sapiens_poses_cb.setChecked(not self.show_sapiens_poses_cb.isChecked()))
        QShortcut(QKeySequence("V"), self, lambda: self.show_invalid_tracks_cb.setChecked(not self.show_invalid_tracks_cb.isChecked()))
        
        # Edit mode shortcuts
        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self.edit_mode_cb.setChecked(not self.edit_mode_cb.isChecked()))
        # E key: if edit mode is active and a track is selected, do engagement correction
        self.e_shortcut = QShortcut(QKeySequence("E"), self)
        self.e_shortcut.activated.connect(self.handle_e_key)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_changes)
        QShortcut(QKeySequence("Ctrl+D"), self, self.discard_changes)
        QShortcut(QKeySequence("C"), self, self.append_checked_flag)
        QShortcut(QKeySequence("Ctrl+F"), self, self.set_track_validity_invalid)
        QShortcut(QKeySequence("Ctrl+X"), self, self.set_engagement_to_zero)
        QShortcut(QKeySequence("Ctrl+W"), self, self.set_engagement_to_three)
        
        # Track selection
        QShortcut(QKeySequence("Escape"), self, self.clear_track_selection)
        
        # Play/Pause
        QShortcut(QKeySequence("Space"), self, self.toggle_play_pause)
        
        # Episode navigation shortcuts
        QShortcut(QKeySequence("Ctrl+B"), self, self.previous_episode)
        # Store next episode shortcut so we can enable/disable it
        self.next_episode_shortcut = QShortcut(QKeySequence("Ctrl+N"), self, self.next_episode)
    
    def load_episode_data(self):
        """Load data for current episode"""
        if not self.episodes_list:
            prWarning("No episodes available")
            return
        
        # Check if we have preloaded data for this episode
        if self.current_episode_idx in self.preloaded_data:
            prInfo(f"Using preloaded data for episode {self.current_episode_idx}")
            preloaded = self.preloaded_data.pop(self.current_episode_idx)
            
            session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
            episode_dir = self.get_episode_dir(session, episode)
            annotations_dir = self.get_annotations_dir(session, episode)
            
            # Store the episodes flags path for this episode
            if self.args.splitted_dataset:
                self.current_episodes_flags_path = os.path.join(self.args.directory, "dataset", session, "episodes_flag.json")
            else:
                self.current_episodes_flags_path = os.path.join(self.args.directory, session, "episodes_flag.json")
            
            # Use preloaded data
            self.detections = preloaded.get('detections', {})
            self.image_files = preloaded.get('image_files', [])
            
            # Extract plot data from detections
            if self.detections:
                detections_list = []
                for frame_idx in sorted(self.detections.keys()):
                    frame_data = self.detections[frame_idx]
                    if "image_file" in frame_data:
                        detections_list.append(frame_data)
                self.extract_plot_data(detections_list)
            
            # Use preloaded video frames if available (mini videos)
            if preloaded.get('video_frames') is not None:
                self.video_frames = preloaded['video_frames']
                self.video_path = preloaded.get('video_path')
                self.video_cap = None
                
                # Set image dimensions from first frame
                if self.video_frames:
                    self.image_height, self.image_width = self.video_frames[0].shape[:2]
                    self.full_image_height = int(self.image_height * self.scale_factor)
                    self.full_image_width = int(self.image_width * self.scale_factor)
                    prInfo(f"Video dimensions: {self.image_width}x{self.image_height} (full: {self.full_image_width}x{self.full_image_height})")
            elif self.video_mode and not self.use_mini:
                # Full resolution: need to open VideoCapture
                video_path = self.get_video_path(session, episode)
                if os.path.exists(video_path):
                    self.video_cap = cv2.VideoCapture(video_path)
                    if self.video_cap.isOpened():
                        ret, first_frame = self.video_cap.read()
                        if ret:
                            self.image_height, self.image_width = first_frame.shape[:2]
                            self.full_image_height = self.image_height
                            self.full_image_width = self.image_width
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Load interact zone tracking
            self.load_interact_zone_tracking(annotations_dir)
            
            # Update UI
            self.update_slider()
            self.current_frame = 0
            self.update_image_display()
            self.update_track_list()
            self.update_report_tracks_list()
            self.update_plots()
            
            # Update interaction UI availability
            has_interaction_data = (self.interaction_mask is not None or self.interact_zone_tracking is not None)
            self.show_interaction_highlights_cb.setEnabled(has_interaction_data)
            self.show_interaction_zone_cb.setEnabled(has_interaction_data)
            
            status_msg = f"Loaded episode {session}/{episode} (from cache) - {len(self.image_files)} frames, {len(self.detections)} detection frames"
            self.statusBar().showMessage(status_msg)
            prInfo(status_msg)
            
            # Initialize edit mode if enabled
            if self.edit_mode:
                self.initialize_edited_detections()
                self.calculate_next_available_id()
            
            # Trigger preload of next episode
            self.preload_next_episode()
            return
            
        # Normal loading path (no preloaded data)
        session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
        episode_dir = self.get_episode_dir(session, episode)
        annotations_dir = self.get_annotations_dir(session, episode)
        
        # Store the episodes flags path for this episode
        if self.args.splitted_dataset:
            self.current_episodes_flags_path = os.path.join(self.args.directory, "dataset", session, "episodes_flag.json")
        else:
            self.current_episodes_flags_path = os.path.join(self.args.directory, session, "episodes_flag.json")
        
        prInfo(f"Loading episode: {session}/{episode} | {associated_episodes_flags_dict[episode]['flag']}")
        
        # Load images or video frames
        if self.video_mode:
            # Load video frames
            video_path = self.get_video_path(session, episode)
            if not os.path.exists(video_path):
                prError(f"Video file not found: {video_path}")
                self.image_files = []
                self.video_cap = None
                self.video_frames = None
            else:
                prInfo(f"Loading video: {video_path}")
                try:
                    # Use OpenCV VideoCapture for reliable video loading
                    self.video_cap = cv2.VideoCapture(video_path)
                    
                    if not self.video_cap.isOpened():
                        prError(f"Failed to open video file: {video_path}")
                        self.image_files = []
                        self.video_cap = None
                        self.video_frames = None
                    else:
                        # Get video properties
                        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        prInfo(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
                        
                        # Create dummy image_files list for compatibility (using frame indices)
                        self.image_files = [f"frame_{i:06d}" for i in range(frame_count)]
                        self.video_path = video_path
                        
                        # For mini videos: load all frames into memory (low memory footprint)
                        # For full resolution: keep VideoCapture open and decode frames on-demand
                        if self.use_mini:
                            prInfo("Loading all frames into memory (mini video - low memory usage)")
                            frames_list = []
                            frame_idx = 0
                            while True:
                                ret, frame = self.video_cap.read()
                                if not ret:
                                    break
                                
                                # OpenCV returns frames in BGR format, which is what we need
                                # Frame is already uint8 numpy array in (H, W, C) format
                                frames_list.append(frame.copy())
                                frame_idx += 1
                                
                                # Progress indicator for long videos
                                if frame_idx % 100 == 0:
                                    prInfo(f"Loaded {frame_idx}/{frame_count} frames...")
                            
                            # Release the VideoCapture as we've loaded all frames
                            self.video_cap.release()
                            self.video_cap = None
                            
                            self.video_frames = frames_list
                            prInfo(f"Loaded {len(self.video_frames)} frames from video into memory")
                            
                            # Set image dimensions from first frame
                            if self.video_frames:
                                self.image_height, self.image_width = self.video_frames[0].shape[:2]
                                self.full_image_height = int(self.image_height * self.scale_factor)
                                self.full_image_width = int(self.image_width * self.scale_factor)
                                prInfo(f"Video dimensions: {self.image_width}x{self.image_height} (full: {self.full_image_width}x{self.full_image_height})")
                        else:
                            # Full resolution: decode frames on-demand to save memory
                            prInfo("Using on-demand frame decoding for full resolution video (saves memory)")
                            
                            # Read first frame to get dimensions
                            ret, first_frame = self.video_cap.read()
                            if ret:
                                self.image_height, self.image_width = first_frame.shape[:2]
                                self.full_image_height = self.image_height
                                self.full_image_width = self.image_width
                                prInfo(f"Video dimensions: {self.image_width}x{self.image_height}")
                                
                                # Reset to beginning of video
                                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            else:
                                prError("Failed to read first frame from video")
                                self.video_cap.release()
                                self.video_cap = None
                                self.image_files = []
                            
                            # Keep video_cap open for on-demand reading
                            # video_frames will be None to indicate on-demand mode
                            self.video_frames = None
                
                except Exception as e:
                    prError(f"Failed to load video: {e}")
                    import traceback
                    traceback.print_exc()
                    self.image_files = []
                    if self.video_cap is not None:
                        self.video_cap.release()
                    self.video_cap = None
                    self.video_frames = None
        
        else:
            # Load images from directory
            if self.use_mini:
                images_dir = os.path.join(episode_dir, "miniatures_images_360_scale_4")
            else:
                images_dir = os.path.join(episode_dir, "images_360")
            
            self.image_files = self.get_image_files(images_dir)
            self.video_cap = None
            self.video_frames = None
        
        # Load detections
        detection_file = os.path.join(annotations_dir, "annotations_360.json")
        if os.path.exists(detection_file):
            detections_list = read_json_to_dic(detection_file)
            detections_list = self.reorder_detections_list(detections_list)
            self.detections = self.convert_detections_list_to_dict(detections_list)
            self.extract_plot_data(detections_list)
            prInfo(f"Loaded {len(detections_list)} detections, {len(self.detections)} detection frames")
        else:
            self.detections = {}
            self.track_engagement_data = {}
            self.track_mask_size_data = {}
            self.track_intersection_pixels_data = {}
            prWarning(f"No detection file found: {detection_file}")
        
        # Load interact zone tracking if available
        self.load_interact_zone_tracking(annotations_dir)
        
        # Update UI
        self.update_slider()
        self.current_frame = 0
        self.update_image_display()
        self.update_track_list()
        self.update_report_tracks_list()
        self.update_plots()
        
        # Update interaction UI availability
        has_interaction_data = (self.interaction_mask is not None or self.interact_zone_tracking is not None)
        self.show_interaction_highlights_cb.setEnabled(has_interaction_data)
        self.show_interaction_zone_cb.setEnabled(has_interaction_data)
        
        status_msg = f"Loaded episode {session}/{episode} - {len(self.image_files)} frames, {len(self.detections)} detection frames"
        self.statusBar().showMessage(status_msg)
        prInfo(status_msg)
        
        # Initialize edit mode if enabled
        if self.edit_mode:
            self.initialize_edited_detections()
            self.calculate_next_available_id()
        
        if self.use_mini:
            # Trigger preload of next episode
            self.preload_next_episode()

    
    def stop_preload_worker(self):
        """Cancel the preload worker (simple cancellation, no waiting)"""
        if self.preload_worker is not None and self.preload_worker.isRunning():
            # Just cancel it - it will finish on its own
            self.preload_worker.cancel()
            self.preload_worker.requestInterruption()
            # Disconnect signals to prevent callbacks
            try:
                self.preload_worker.finished.disconnect()
                self.preload_worker.error.disconnect()
            except:
                pass
            # Don't wait - just let it finish naturally
            self.preload_worker = None
    
    def preload_next_episode(self):
        """Preload next episode's data in background"""
        # Stop any existing preload worker
        self.stop_preload_worker()
        
        # Check if there's a next episode
        next_idx = self.current_episode_idx + 1
        if next_idx >= len(self.episodes_list):
            # No next episode - disable button
            if hasattr(self, 'next_episode_btn'):
                self.next_episode_btn.setEnabled(False)
            return  # No next episode
        
        # Check if already preloaded
        if next_idx in self.preloaded_data:
            # Already preloaded - enable button
            if hasattr(self, 'next_episode_btn'):
                self.next_episode_btn.setEnabled(True)
            return  # Already preloaded
        
        # Clean up old preloaded data (keep only current and next)
        keys_to_remove = [k for k in self.preloaded_data.keys() if k != self.current_episode_idx and k != next_idx]
        for k in keys_to_remove:
            preloaded = self.preloaded_data.pop(k)
            # Clean up video frames if present
            if preloaded.get('video_frames') is not None:
                del preloaded['video_frames']
        
        # Disable next episode button while preloading
        if hasattr(self, 'next_episode_btn'):
            self.next_episode_btn.setEnabled(False)
            self.next_episode_btn.setText("Next ▶ [Loading...]")
        
        # Start preload worker
        prInfo(f"Preloading episode {next_idx} in background...")
        self.preload_worker = EpisodePreloadWorker(self, next_idx)
        self.preload_worker.finished.connect(self.on_preload_finished)
        self.preload_worker.error.connect(self.on_preload_error)
        self.preload_worker.start()
    
    def on_preload_finished(self, episode_idx, detections_dict, video_frames, video_cap, video_path, image_files):
        """Handle preload completion"""
        prInfo(f"Preload completed for episode {episode_idx}")
        self.preloaded_data[episode_idx] = {
            'detections': detections_dict,
            'video_frames': video_frames,
            'video_cap': video_cap,
            'video_path': video_path,
            'image_files': image_files
        }
        self.preload_worker = None
        
        # Re-enable next episode button if this was the next episode
        if hasattr(self, 'next_episode_btn'):
            next_idx = self.current_episode_idx + 1
            if episode_idx == next_idx:
                self.next_episode_btn.setEnabled(True)
                self.next_episode_btn.setText("Next ▶ [Ctrl+N]")
            self.update_next_episode_button_state()
    
    def on_preload_error(self, episode_idx, error_msg):
        """Handle preload error"""
        prWarning(f"Preload failed for episode {episode_idx}: {error_msg}")
        self.preload_worker = None
        
        # Re-enable next episode button even on error
        if hasattr(self, 'next_episode_btn'):
            self.update_next_episode_button_state()
    
    def update_next_episode_button_state(self):
        """Update the next episode button, shortcut, and dropdown enabled state"""
        if not hasattr(self, 'next_episode_btn'):
            return
        
        next_idx = self.current_episode_idx + 1
        has_next = next_idx < len(self.episodes_list)
        is_preloading = self.preload_worker is not None and self.preload_worker.isRunning()
        is_preloaded = next_idx in self.preloaded_data if has_next else False
        
        # Enable if there's a next episode and either preloaded or not currently preloading
        enabled = has_next and (is_preloaded or not is_preloading)
        self.next_episode_btn.setEnabled(enabled)
        
        # Also enable/disable the Ctrl+N shortcut
        if hasattr(self, 'next_episode_shortcut'):
            self.next_episode_shortcut.setEnabled(enabled)
        
        # Disable episode dropdown during preloading to prevent switching episodes
        if hasattr(self, 'episode_combo'):
            self.episode_combo.setEnabled(not is_preloading)
        
        if is_preloading:
            self.next_episode_btn.setText("Next ▶ [Loading...]")
        else:
            self.next_episode_btn.setText("Next ▶ [Ctrl+N]")

    
    def get_image_files(self, directory):
        """Get sorted list of image files from directory"""
        if not os.path.exists(directory):
            prWarning(f"Image directory does not exist: {directory}")
            return []
        
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(os.path.join(root, filename))
        
        return sorted(files)
    
    def reorder_detections_list(self, detections_list):
        """Reorder detections list based on the image_file basenames, it should be done but sometimes it's not"""
        # reorder based on the image_file basenames
        detections_list_basenames = [os.path.basename(detection_entry["image_file"]) for detection_entry in detections_list]
        detections_list_basenames_argsort = np.argsort(detections_list_basenames)
        return [detections_list[i] for i in detections_list_basenames_argsort]
    
    def convert_detections_list_to_dict(self, detections_list):
        """Convert detections list format to indexed dictionary format"""
        detections_dict = {}
        
        assert(len(detections_list) == len(self.image_files)), "Number of detections and images should be the same"
        
        for i, (detection_entry, image_file) in enumerate(zip(detections_list, self.image_files)):
            det_image_file = detection_entry["image_file"]
            
            if os.path.basename(image_file).startswith("frame_"):
                # no check in this case, these are just ordered frames extracted from the video, before renaming, a bit more risky but it should be fine since the detections in the annotations should also be ordered
                pass
            else:
                assert(os.path.basename(image_file) == os.path.basename(det_image_file)), f"Image file: {os.path.basename(image_file)} vs detection: {os.path.basename(det_image_file)}"
            
            detections_dict[i] = {
                "image_file": detection_entry["image_file"],
                "detections": detection_entry["detections"]
            }
        
        return detections_dict
    
    def extract_plot_data(self, detections_list):
        """Extract engagement and mask size data for plots"""
        self.track_engagement_data = {}
        self.track_mask_size_data = {}
        self.track_intersection_pixels_data = {}
        
        for i, detection_entry in enumerate(detections_list):
            if True:
                frame_index = i
                
                for detection in detection_entry["detections"]:
                    track_id = detection["track_id"]
                    
                    # Extract engagement level if available
                    engagement_level = detection.get("engagement_level", -1)
                    if engagement_level >= 0:  # Valid engagement level
                        if track_id not in self.track_engagement_data:
                            self.track_engagement_data[track_id] = {'frames': [], 'engagement': []}
                        self.track_engagement_data[track_id]['frames'].append(frame_index)
                        self.track_engagement_data[track_id]['engagement'].append(engagement_level)
                    
                    # Extract intersection pixels if available
                    intersection_pixels = detection.get("intersection_pixels", -1)
                    if intersection_pixels >= 0:  # Valid intersection pixels
                        if track_id not in self.track_intersection_pixels_data:
                            self.track_intersection_pixels_data[track_id] = {'frames': [], 'intersection_pixels': []}
                        self.track_intersection_pixels_data[track_id]['frames'].append(frame_index)
                        self.track_intersection_pixels_data[track_id]['intersection_pixels'].append(intersection_pixels)
                    
                    # Extract mask size if RLE is available
                    if "mask_rle" in detection:
                        mask_rle = [int(x) for x in detection["mask_rle"].split(",")]
                        mask_rle = torch.tensor(mask_rle, dtype=torch.int32)
                        # Calculate mask size by decoding RLE and counting pixels
                        # For efficiency, we can estimate from RLE without full decoding
                        if "mask_size" in detection:
                            mask_size = detection["mask_size"]
                        else:
                            # fallback for older versions
                            mask_size = self.estimate_mask_size_from_rle(mask_rle)
                        
                        if track_id not in self.track_mask_size_data:
                            self.track_mask_size_data[track_id] = {'frames': [], 'mask_size': []}
                        self.track_mask_size_data[track_id]['frames'].append(frame_index)
                        self.track_mask_size_data[track_id]['mask_size'].append(mask_size)

    def estimate_mask_size_from_rle(self, rle_tensor):
        """Estimate mask size from RLE encoding without full decoding"""
        # RLE format: [starts_with_true, run1, run2, run3, ...]
        # Positive pixels are in alternating runs starting from starts_with_true
        starts_with_true = rle_tensor[0].item()
        runs = rle_tensor[1:]
        
        total_positive = 0
        for i, run_length in enumerate(runs):
            if (starts_with_true and i % 2 == 0) or (not starts_with_true and i % 2 == 1):
                total_positive += run_length.item()
        
        return total_positive
    
    def get_mask_intersection_with_interaction_zone(self, mask_rle, image_shape, frame_index=None):
        """Get the intersection of a track mask with the interaction zone"""
        # Get the appropriate interaction mask for this frame
        if frame_index is not None:
            interaction_mask_resized, _ = self.get_interaction_mask_for_frame(frame_index, image_shape)
        else:
            interaction_mask_resized = self.get_static_interaction_mask(image_shape)
            
        if interaction_mask_resized is None:
            return None
    
        # Decode the mask
        mask = decode_RLE(mask_rle.to(MASK_RLE_DEVICE), image_shape)
        mask_np = mask.cpu().numpy().astype(bool)
        
        # Find intersection
        intersection = mask_np & interaction_mask_resized
        
        return intersection

    def get_interaction_mask_for_frame(self, frame_index, image_shape):
        """Get the interaction mask for a specific frame (dynamic or static)"""
        if self.interact_zone_tracking is not None:
            # Use dynamic interaction zone tracking
            if frame_index < len(self.interact_zone_tracking):
                interact_zone_entry = self.interact_zone_tracking[frame_index]
                if "detections" in interact_zone_entry and len(interact_zone_entry["detections"]) > 0:
                    # Get the first detection as interaction zone mask
                    interaction_det = interact_zone_entry["detections"][0]
                    if "mask_rle" in interaction_det:
                        mask_rle = [int(x) for x in interaction_det["mask_rle"].split(",")]
                        mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                        mask = decode_RLE(mask_rle, image_shape)
                        bbox = [interaction_det["xmin"], interaction_det["ymin"], interaction_det["xmax"], interaction_det["ymax"]]
                        return mask.cpu().numpy().astype(bool), bbox

            # Fallback to static mask if dynamic fails or not available
            return self.get_static_interaction_mask(image_shape), None
        else:
            # Use static interaction mask
            return self.get_static_interaction_mask(image_shape), None
    
    def get_static_interaction_mask(self, image_shape):
        """Get the static interaction mask resized to image shape"""
        if self.interaction_mask is None:
            return None
            
        # Resize interaction mask to match image size if needed
        if self.interaction_mask.shape != image_shape:
            interaction_mask_resized = cv2.resize(
                self.interaction_mask.astype(np.uint8), 
                (image_shape[1], image_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            interaction_mask_resized = self.interaction_mask
            
        return interaction_mask_resized
    
    def update_slider(self):
        """Update slider range based on loaded data"""
        if self.image_files:
            max_val = len(self.image_files) - 1
            self.frame_slider.setRange(0, max_val)
            self.frame_slider.setValue(0)
            self.frame_label.setText(f"0/{max_val}")
        else:
            self.frame_slider.setRange(0, 0)
            self.frame_label.setText("0/0")
    
    def on_slider_change(self, value):
        """Handle slider value change"""
        self.current_frame = value
        self.update_image_display()
        self.update_track_list()
        self.update_plots()
        self.frame_label.setText(f"{value}/{len(self.image_files) - 1}")
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.frame_slider.setValue(self.current_frame - 1)
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < len(self.image_files) - 1:
            self.frame_slider.setValue(self.current_frame + 1)
    
    def prev_frame_10(self):
        """Go back 10 frames"""
        new_frame = max(0, self.current_frame - 10)
        self.frame_slider.setValue(new_frame)
    
    def next_frame_10(self):
        """Go forward 10 frames"""
        new_frame = min(len(self.image_files) - 1, self.current_frame + 10)
        self.frame_slider.setValue(new_frame)
    
    def update_image_display(self):
        tic_global = time.time()
        """Update image display with tracks"""
        if not self.image_files or self.current_frame >= len(self.image_files):
            return
        
        # Use edited detections if in edit mode and available
        if self.edit_mode and self.current_frame in self.edited_detections:
            detections = self.edited_detections[self.current_frame]["detections"]
        else:
            detections = self.detections[self.current_frame]["detections"] if self.current_frame in self.detections else []
        
        # Load image or get frame from video
        tic = time.time()
        if self.video_mode:
            # Get frame from video
            if self.video_frames is not None and len(self.video_frames) > 0:
                # Mini video: frames are pre-loaded in memory
                if self.current_frame >= len(self.video_frames):
                    prError(f"Video frame {self.current_frame} not available")
                    return
                image = self.video_frames[self.current_frame].copy()
            elif self.video_cap is not None:
                # Full resolution: decode frame on-demand
                if self.current_frame >= len(self.image_files):
                    prError(f"Video frame {self.current_frame} not available")
                    return
                
                # Seek to the requested frame
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, image = self.video_cap.read()
                if not ret:
                    prError(f"Failed to read frame {self.current_frame} from video")
                    return
            else:
                prError("Video not loaded properly")
                return
        else:
            # Load image from file
            image_file = self.image_files[self.current_frame]
            image = cv2.imread(image_file)
            if image is None:
                prError(f"Could not load image: {image_file}")
                return
        
        # Store image dimensions (miniature size if using mini)
        # Only update if not already set from video loading
        if not self.video_mode or self.image_height is None:
            self.image_height, self.image_width = image.shape[:2]
        
        # Store full image dimensions for mask decoding (if not set yet)
        if self.use_mini and (self.full_image_height is None or self.full_image_width is None):
            # Full size is scale_factor times the miniature size
            self.full_image_height = int(self.image_height * self.scale_factor)
            self.full_image_width = int(self.image_width * self.scale_factor)
            prInfo(f"Using miniature images: {self.image_width}x{self.image_height} -> Full size: {self.full_image_width}x{self.full_image_height}")
        elif not self.use_mini and (self.full_image_height is None or self.full_image_width is None):
            # Not using mini, so current size is full size
            self.full_image_height = self.image_height
            self.full_image_width = self.image_width
        
        tac = time.time()
        # prTimer(f"Load image {image_file}", tic, tac)
        
        # Draw tracks on the image
        tic = time.time()
        image = self.draw_tracks_on_image(image, detections)
        tac = time.time()
        # prTimer(f"Draw tracks on image", tic, tac)
        
        # Calculate and apply horizontal roll to center interaction mask
        image = self.apply_horizontal_roll(image)
        
        # Convert BGR to RGB for Qt
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image to widget
        self.image_widget.set_image_with_scale(image_rgb)
        
        # Update image info
        info_text = f"Frame {self.current_frame} - {len(detections)} tracks"
        self.info_label.setText(info_text)
        tac_global = time.time()
        # prTimer(f"Update image display", tic_global, tac_global)
    
    def apply_horizontal_roll(self, image):
        """Apply horizontal roll to center the interaction mask"""
        if not self.enable_image_rolling:
            self.horizontal_roll_offset = 0
            return image
        
        image_height, image_width = image.shape[:2]
        
        # Get interaction mask bbox for current frame
        full_image_height = self.full_image_height if self.use_mini else image_height
        full_image_width = self.full_image_width if self.use_mini else image_width
        
        interaction_mask, bbox = self.get_interaction_mask_for_frame(self.current_frame, (full_image_height, full_image_width))
        
        if bbox is None:
            # No bbox available, don't roll
            self.horizontal_roll_offset = 0
            return image
        
        # Calculate bbox center (handle 360° wrapping)
        xmin, ymin, xmax, ymax = bbox
        
        # Scale bbox to image size if using mini
        if self.use_mini:
            xmin = int(xmin / self.scale_factor)
            ymin = int(ymin / self.scale_factor)
            xmax = int(xmax / self.scale_factor)
            ymax = int(ymax / self.scale_factor)
        
        # # Calculate center x coordinate (handle wrapping)
        # # For 360° images, we need to handle wrapping carefully
        # if xmax > image_width:
        #     # Bbox wraps around
        #     if xmin > image_width:
        #         # Entire bbox is in wrapped part (xmin and xmax both > image_width)
        #         # Unwrap coordinates
        #         unwrapped_xmin = xmin - image_width
        #         unwrapped_xmax = xmax - image_width
        #         center_x = (unwrapped_xmin + unwrapped_xmax) / 2.0
        #     else:
        #         # Bbox spans both sides (xmin < image_width, xmax > image_width)
        #         # The bbox spans from xmin to image_width (left part) and from 0 to (xmax - image_width) (right part)
        #         left_part_width = image_width - xmin
        #         right_part_width = xmax - image_width
                
        #         # Calculate center based on which part is larger
        #         if left_part_width >= right_part_width:
        #             # Center is in the left part
        #             center_x = xmin + left_part_width / 2.0
        #         else:
        #             # Center is in the right part (wrapped)
        #             center_x = right_part_width / 2.0
        # else:
        # Normal bbox, no wrapping
        center_x = (xmin + xmax) / 2.0
        
        # Calculate offset to center the bbox center at image_width/2
        target_center = image_width / 2.0
        offset = int(target_center - center_x)
        
        # Handle wrapping: if offset is large, we might want to roll the other way
        # For 360° images, we can roll in either direction
        if abs(offset) > image_width / 2:
            # Roll the other way (shorter distance)
            if offset > 0:
                offset = offset - image_width
            else:
                offset = offset + image_width
        
        # Store offset for click remapping
        self.horizontal_roll_offset = offset
        
        # Apply horizontal roll using numpy roll
        if offset != 0:
            image = np.roll(image, offset, axis=1)
        
        return image
    
    def draw_tracks_on_image(self, image, detections):
        """Draw track visualizations on image"""
        image_copy = image.copy()
        image_height, image_width = image.shape[:2]
        
        # Use full image dimensions for mask decoding and interaction zone
        full_image_height = self.full_image_height if self.use_mini else image_height
        full_image_width = self.full_image_width if self.use_mini else image_width
        
        # Draw interaction zone as background overlay if enabled
        if self.show_interaction_zone:
            interaction_mask, _ = self.get_interaction_mask_for_frame(self.current_frame, (full_image_height, full_image_width))
            if interaction_mask is not None and self.use_mini:
                # Scale down interaction mask to miniature size
                interaction_mask = cv2.resize(interaction_mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if interaction_mask is not None:
                # Create semi-transparent overlay for interaction zone
                zone_color = (255, 255, 0)  # Yellow for interaction zone
                overlay = image_copy.copy()
                overlay[interaction_mask] = zone_color
                alpha = 0.2  # 20% opacity for zone display
                image_copy = cv2.addWeighted(image_copy, 1.0-alpha, overlay, alpha, 0)
                
                # Add contour around interaction zone
                zone_uint8 = interaction_mask.astype(np.uint8)
                contours, _ = cv2.findContours(zone_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(image_copy, contours, -1, zone_color, 2)
        
        for det in detections:
            track_id = det.get("track_id", -1)
            engagement_level = det.get("engagement_level", det.get("engagement_level_auto", -1)) # where the fuck does this engagement_level_auto come from... 
            intersection_pixels = det.get("intersection_pixels", -1)
            
            # Filter based on track validity if option is disabled
            if not self.show_invalid_tracks:
                validity = det.get("validity", "valid")
                if validity != "valid":
                    continue
            
            # Get color
            color = self.get_track_color(track_id)
            
            # Draw bounding box if enabled
            if self.show_boxes:
                thickness = 2
                if self.is_track_selected(track_id):
                    thickness = 4  # Thicker for selected track
                    
                # Scale coordinates down if using mini images
                xmin = int(det["xmin"] / self.scale_factor)
                ymin = int(det["ymin"] / self.scale_factor)
                xmax = int(det["xmax"] / self.scale_factor)
                ymax = int(det["ymax"] / self.scale_factor)
                
                if xmax > image_copy.shape[1]:
                    if xmin > image_copy.shape[1]:
                        # fully outside
                        cv2.rectangle(image_copy, (xmin-image_copy.shape[1], ymin), (xmax-image_copy.shape[1], ymax), color, thickness)
                    else:
                        # two sides
                        cv2.rectangle(image_copy, (xmin, ymin), (image_copy.shape[1], ymax), color, thickness)
                        cv2.rectangle(image_copy, (0, ymin), (xmax-image_copy.shape[1], ymax), color, thickness)
                else:
                    cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Draw track ID if enabled
            if self.show_track_ids:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35 if self.use_mini else 0.7
                font_thickness = 2
                
                text = f"ID: {track_id}"
                x1 = int(det["xmin"] / self.scale_factor)
                if x1 > image_copy.shape[1]:
                    x1 = x1 - image_copy.shape[1]
                y1 = int(det["ymin"] / self.scale_factor)
                
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0], y1), color, -1)
                cv2.putText(image_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
            
            # Draw mask if enabled and available
            if self.show_masks and "mask_rle" in det:
                # Decode RLE mask to full size first
                mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                mask = decode_RLE(mask_rle, (full_image_height, full_image_width))
                mask_np = mask.cpu().numpy().astype(bool)
                
                # Scale down to miniature size if using mini
                if self.use_mini:
                    mask_np = cv2.resize(mask_np.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # Apply colored overlay
                overlay = image_copy.copy()
                overlay[mask_np] = color
                alpha = self.mask_opacity / 100.0
                image_copy = cv2.addWeighted(image_copy, 1.0-alpha, overlay, alpha, 0)

            # Draw interaction highlights for engagement level 3 tracks
            if (self.show_interaction_highlights and 
                (engagement_level == 3 or intersection_pixels > 0) and 
                "mask_rle" in det and 
                self.interaction_mask is not None):
                mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                mask_rle = torch.tensor(mask_rle, dtype=torch.int32)
                intersection = self.get_mask_intersection_with_interaction_zone(
                    mask_rle, (full_image_height, full_image_width), self.current_frame)
                
                # Scale down intersection mask if using mini
                if intersection is not None and self.use_mini:
                    intersection = cv2.resize(intersection.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                if intersection is not None and intersection.any():
                    # Create a bright highlight overlay for interaction areas
                    highlight_color = (0, 255, 255)  # Bright cyan for interaction zones
                    overlay = image_copy.copy()
                    overlay[intersection] = highlight_color
                    
                    # Apply strong highlight with higher opacity
                    alpha = 0.7  # 70% opacity for interaction highlights
                    image_copy = cv2.addWeighted(image_copy, 1.0-alpha, overlay, alpha, 0)
                    
                    # Add contour around interaction area
                    intersection_uint8 = intersection.astype(np.uint8)
                    contours, _ = cv2.findContours(intersection_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(image_copy, contours, -1, highlight_color, 3)

            # Draw thick contour if this track is selected
            if self.is_track_selected(track_id):
                if self.show_masks and "mask_rle" in det:
                    # Use mask contour for selected track
                    mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                    mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                    mask = decode_RLE(mask_rle, (full_image_height, full_image_width))
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    
                    # Scale down mask if using mini
                    if self.use_mini:
                        mask_np = cv2.resize(mask_np, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Draw the largest contour with thick line
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(image_copy, [largest_contour], -1, color, 6)
                else:
                    # Fall back to thick bounding box when masks not available
                    x1, y1, x2, y2 = int(det["xmin"] / self.scale_factor), int(det["ymin"] / self.scale_factor), int(det["xmax"] / self.scale_factor), int(det["ymax"] / self.scale_factor)
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 8)
            
            # Draw ViTPose keypoints and skeleton if enabled
            if self.show_vitpose_poses:
                # Draw ViTPose keypoints if available
                scores_vitpose = det.get("scores_vitpose", [0.0]*17)
                if len(scores_vitpose) >= 17 and np.array(scores_vitpose).max() > self.vitpose_min_score:
                    keypoints = det.get("keypoints_vitpose", [])
                    if len(keypoints) >= 34:  # 17 keypoints * 2 coordinates
                        keypoints = np.array(keypoints).reshape(17, 2)
                        
                        # Choose pose color based on mask opacity
                        pose_color = (255, 255, 255) if self.mask_opacity > 80 else color  # White if high opacity
                        
                        # Draw keypoints
                        point_radius = 3 if self.use_mini else 10
                        for i in range(keypoints.shape[0]):
                            if i < len(scores_vitpose) and scores_vitpose[i] > self.vitpose_min_score:
                                kp_x, kp_y = int(keypoints[i, 0] / self.scale_factor), int(keypoints[i, 1] / self.scale_factor)
                                if kp_x > 0 and kp_y > 0:
                                    cv2.circle(image_copy, (kp_x, kp_y), point_radius, pose_color, -1)
                        
                        # Draw skeleton connections
                        line_thickness = 1 if self.use_mini else 2
                        for edge in COCO_EDGES:
                            if (edge[0] < len(scores_vitpose) and edge[1] < len(scores_vitpose) and 
                                scores_vitpose[edge[0]] > self.vitpose_min_score and scores_vitpose[edge[1]] > self.vitpose_min_score):
                                pt1 = (int(keypoints[edge[0], 0] / self.scale_factor), int(keypoints[edge[0], 1] / self.scale_factor))
                                pt2 = (int(keypoints[edge[1], 0] / self.scale_factor), int(keypoints[edge[1], 1] / self.scale_factor))
                                # put a maximum distance to avoid cross image points
                                if abs(pt1[0] - pt2[0]) > image_width // 4:
                                    continue
                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    cv2.line(image_copy, pt1, pt2, pose_color, line_thickness)
            
            # Draw Sapiens keypoints and skeleton if enabled
            if self.show_sapiens_poses:
                # Draw Sapiens keypoints if available
                det_keys = list(det.keys())
                det_keys_sapiens = [key for key in det_keys if key.startswith("scores_sapiens")] # scores_sapiens_308_03b for example (scores_sapiens_NKPTS_MODEL)
                if len(det_keys_sapiens) > 0:
                    sapiens_model = det_keys_sapiens[0].split("_")[-1] # if several possibility always take the first one
                    sapiens_num_keypoints = int(det_keys_sapiens[0].split("_")[2]) # if several possibility always take the first one
                    assert sapiens_num_keypoints == 308, "Sapiens number of keypoints {} not supported (only 308 for now)".format(sapiens_num_keypoints)
                    sapiens_score_key = f"scores_sapiens_{sapiens_num_keypoints}_{sapiens_model}"
                    sapiens_keypoints_key = f"keypoints_sapiens_{sapiens_num_keypoints}_{sapiens_model}"
                    sapiens_scores = det.get(sapiens_score_key, [0.0]*sapiens_num_keypoints)
                    sapiens_keypoints = det.get(sapiens_keypoints_key, [])
                    if len(sapiens_scores) >= sapiens_num_keypoints and np.array(sapiens_scores).max() > self.sapiens_min_score:
                        sapiens_keypoints = np.array(sapiens_keypoints).reshape(sapiens_num_keypoints, 2)
                        
                        # Draw keypoints
                        point_radius = 1 if self.use_mini else 3
                        for i in range(sapiens_keypoints.shape[0]):
                            if sapiens_scores[i] > self.sapiens_min_score:
                                kpt_color = GOLIATH_KPTS_COLORS[i]
                                kp_x, kp_y = int(sapiens_keypoints[i, 0] / self.scale_factor), int(sapiens_keypoints[i, 1] / self.scale_factor)
                                if kp_x > 0 and kp_y > 0:
                                    cv2.circle(image_copy, (kp_x, kp_y), point_radius, kpt_color, -1)
                                
                        # Draw skeleton connections
                        line_thickness = 1 if self.use_mini else 2
                        for edge_idx in GOLIATH_SKELETON_INFO:
                            edge = GOLIATH_SKELETON_INFO[edge_idx]
                            kp1_score = sapiens_scores[edge["link"][0]]
                            kp2_score = sapiens_scores[edge["link"][1]]
                            if kp1_score > self.sapiens_min_score and kp2_score > self.sapiens_min_score:
                                edge_color = edge["color"]
                                kp_x1, kp_y1 = int(sapiens_keypoints[edge["link"][0], 0] / self.scale_factor), int(sapiens_keypoints[edge["link"][0], 1] / self.scale_factor)
                                kp_x2, kp_y2 = int(sapiens_keypoints[edge["link"][1], 0] / self.scale_factor), int(sapiens_keypoints[edge["link"][1], 1] / self.scale_factor)
                                if abs(kp_x1 - kp_x2) > image_width // 4:
                                    continue
                                if kp_x1 > 0 and kp_y1 > 0 and kp_x2 > 0 and kp_y2 > 0:
                                    cv2.line(image_copy, (kp_x1, kp_y1), (kp_x2, kp_y2), edge_color, line_thickness)
        
        return image_copy
    
    def update_track_list(self):
        """Update the track list widget"""
        self.track_list.clear()
        
        # Use edited detections if in edit mode and available
        if self.edit_mode and self.current_frame in self.edited_detections:
            detections = self.edited_detections[self.current_frame]["detections"]
        elif self.current_frame in self.detections:
            detections = self.detections[self.current_frame]["detections"]
        else:
            return
        
        for det in detections:
            track_id = det.get("track_id", -1)
            validity = det.get("validity", "valid")
            
            # Filter based on track validity if option is disabled
            if not self.show_invalid_tracks:
                if validity != "valid":
                    continue
            
            # Show validity status in track list
            if self.edit_mode:
                track_text = f"Track {track_id} ({validity})"
            else:
                track_text = f"Track {track_id}"
            color = self.get_track_color(track_id)
            
            item = QListWidgetItem(track_text)
            
            # Set background color to match track color
            rgb_color = (int(min(color[2]*1.5, 255)), int(min(color[1]*1.5, 255)), int(min(color[0]*1.5, 255)))
            item.setBackground(QColor(*rgb_color))
            
            # Set text color based on background brightness
            brightness = sum(rgb_color) / 3
            text_color = QColor(255, 255, 255) if brightness < 128 else QColor(0, 0, 0)
            item.setForeground(text_color)
            
            # Store track_id as user data
            item.setData(Qt.ItemDataRole.UserRole, track_id)
            
            self.track_list.addItem(item)
    
    def update_report_tracks_list(self):
        """Update the report tracks list for the current episode"""
        if self.report_tracks_list is None:
            return
        
        self.report_tracks_list.clear()
        
        if self.report_data is None or not self.episodes_list:
            return
        
        # Get current episode
        if self.current_episode_idx >= len(self.episodes_list):
            return
        
        session, episode, _ = self.episodes_list[self.current_episode_idx]
        episode_key = (session, episode)
        
        # Get tracks for this episode from report
        tracks = self.report_tracks_by_episode.get(episode_key, [])
        
        # Sort tracks by first frame
        tracks_sorted = sorted(tracks, key=lambda x: x['first_frame'] if x['first_frame'] is not None else float('inf'))
        
        # Status color mapping
        status_colors = {
            'tp': '#00AA00',  # Green
            'fp': '#FF0000',  # Red
            'tn': '#888888',  # Gray
            'fn': '#FF8800',  # Orange
            'unknown': '#000000'  # Black
        }
        
        for track_info in tracks_sorted:
            unique_track_id = track_info['unique_track_id']
            status = track_info['status']
            first_frame = track_info['first_frame']
            last_frame = track_info['last_frame']
            predicted_score = track_info.get('predicted_score', 0.0)
            
            # Extract track ID from unique_track_id (format: recording_episode_trackid)
            track_id_str = unique_track_id.rsplit('_', 1)[-1] if '_' in unique_track_id else unique_track_id
            
            # Create display text
            if first_frame is not None and last_frame is not None:
                frame_text = f"Frames: {first_frame}-{last_frame}"
            elif first_frame is not None:
                frame_text = f"Frame: {first_frame}"
            else:
                frame_text = "Frames: N/A"
            
            display_text = f"[{status.upper()}] Track {track_id_str} | {frame_text} | Score: {predicted_score:.3f}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, track_info)  # Store full track info
            
            # Set color based on status
            color = QColor(status_colors.get(status, '#000000'))
            item.setForeground(QBrush(color))
            
            self.report_tracks_list.addItem(item)
    
    def on_report_track_double_clicked(self, item):
        """Handle double-click on report track - jump to first frame"""
        track_info = item.data(Qt.ItemDataRole.UserRole)
        if track_info is None:
            return
        
        first_frame = track_info.get('first_frame')
        if first_frame is not None and 0 <= first_frame < len(self.image_files):
            self.frame_slider.setValue(first_frame)
            self.update_image_display()
            prInfo(f"Jumped to frame {first_frame} for track {track_info.get('unique_track_id', 'unknown')}")
    
    def on_track_selected(self, item):
        """Handle track selection from the list"""
        # When using list widget selection, clear single selection
        # The list widget handles multi-selection automatically
        self.selected_track = None
        self.update_image_display()
    
    def clear_track_selection(self):
        """Clear the current track selection"""
        self.selected_track = None
        self.track_list.clearSelection()
        self.update_image_display()
    
    def select_track_by_click(self, image_x, image_y, ctrl_pressed=False):
        """Select a track by clicking on it in the image
        
        Args:
            image_x: X coordinate in image space (miniature size if using mini)
            image_y: Y coordinate in image space (miniature size if using mini)
            ctrl_pressed: If True, toggle track in multi-selection. If False, single select.
        """
        if not self.edit_mode:
            return
        
        # Scale up click coordinates to full size for comparison with detection coordinates
        full_image_x = int(image_x * self.scale_factor)
        full_image_y = int(image_y * self.scale_factor)
        
        # Get current detections
        if self.edit_mode and self.current_frame in self.edited_detections:
            detections = self.edited_detections[self.current_frame]["detections"]
        elif self.current_frame in self.detections:
            detections = self.detections[self.current_frame]["detections"]
        else:
            return
        
        # Find detections whose bounding boxes contain the click point
        matching_dets = []
        for det in detections:
            xmin, ymin, xmax, ymax = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
            
            # Handle 360° wrapping - boxes can wrap around (use full image width for comparison)
            full_image_width = self.full_image_width if self.use_mini else self.image_width
            if xmax > full_image_width:
                # Box wraps around - check if click is in either part
                if ((xmin <= full_image_x < full_image_width) or (0 <= full_image_x < xmax - full_image_width)) and ymin <= full_image_y < ymax:
                    matching_dets.append(det)
            else:
                # Normal box
                if xmin <= full_image_x < xmax and ymin <= full_image_y < ymax:
                    matching_dets.append(det)
        
        if not matching_dets:
            # No bounding box matches
            if not ctrl_pressed:
                # Clear selection only if not Ctrl+click (Ctrl+click on empty space should do nothing)
                self.clear_track_selection()
            return
        
        # If only one match, select it
        if len(matching_dets) == 1:
            track_id = matching_dets[0].get("track_id", -1)
            if ctrl_pressed:
                # Toggle track in list selection
                self.toggle_track_in_list_selection(track_id)
            else:
                # Single select
                self.selected_track = track_id
                self.update_track_selection_in_list()
            self.update_image_display()
            return
        
        # Multiple boxes match - check masks
        mask_matching_dets = []
        for det in matching_dets:
            if "mask_rle" in det:
                try:
                    # Decode mask to full size
                    mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                    mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                    full_height = self.full_image_height if self.use_mini else self.image_height
                    full_width = self.full_image_width if self.use_mini else self.image_width
                    mask = decode_RLE(mask_rle, (full_height, full_width))
                    mask_np = mask.cpu().numpy().astype(bool)
                    
                    # Check if click point is in mask (use full-size coordinates)
                    if 0 <= full_image_y < mask_np.shape[0] and 0 <= full_image_x < mask_np.shape[1]:
                        if mask_np[full_image_y, full_image_x]:
                            mask_matching_dets.append(det)
                except Exception as e:
                    prWarning(f"Error decoding mask for track selection: {e}")
                    # If mask decode fails, include it anyway (fallback to box)
                    mask_matching_dets.append(det)
            else:
                # No mask available, use bounding box as fallback
                mask_matching_dets.append(det)
        
        # Select the first matching track (or first box match if no mask matches)
        if mask_matching_dets:
            selected_det = mask_matching_dets[0]
        else:
            selected_det = matching_dets[0]
        
        track_id = selected_det.get("track_id", -1)
        if ctrl_pressed:
            # Toggle track in list selection
            self.toggle_track_in_list_selection(track_id)
        else:
            # Single select
            self.selected_track = track_id
            self.update_track_selection_in_list()
        self.update_image_display()
    
    def toggle_track_in_list_selection(self, track_id):
        """Toggle a track in the list widget's selection"""
        # Find the item in the list
        for i in range(self.track_list.count()):
            item = self.track_list.item(i)
            item_track_id = item.data(Qt.ItemDataRole.UserRole)
            if item_track_id == track_id:
                # Toggle selection
                if item.isSelected():
                    item.setSelected(False)
                else:
                    item.setSelected(True)
                self.track_list.scrollToItem(item)
                # Clear single selection if we're using multi-selection
                self.selected_track = None
                return
    
    def update_track_selection_in_list(self):
        """Update the track list widget to reflect the selected track"""
        if self.selected_track is None:
            # If no single selection, keep current list selection
            return
        
        # Clear current selection and select only the single track
        self.track_list.clearSelection()
        
        # Find and select the item in the list
        for i in range(self.track_list.count()):
            item = self.track_list.item(i)
            track_id = item.data(Qt.ItemDataRole.UserRole)
            if track_id == self.selected_track:
                item.setSelected(True)
                self.track_list.setCurrentItem(item)
                self.track_list.scrollToItem(item)
                return
    
    def update_plots(self):
        """Update both engagement and mask size plots"""
        if not hasattr(self, 'engagement_plot') or not hasattr(self, 'mask_size_plot') or not hasattr(self, 'intersection_pixels_plot'):
            return
        
        # Prepare intersection pixels data for current window
        intersection_pixels_data = {}
        for track_id, data in self.track_intersection_pixels_data.items():
            frame_value_pairs = list(zip(data['frames'], data['intersection_pixels']))
            intersection_pixels_data[track_id] = frame_value_pairs
        
        # Prepare engagement data for current window
        engagement_data = {}
        for track_id, data in self.track_engagement_data.items():
            frame_value_pairs = list(zip(data['frames'], data['engagement']))
            engagement_data[track_id] = frame_value_pairs
        
        # Prepare mask size data for current window
        mask_size_data = {}
        for track_id, data in self.track_mask_size_data.items():
            frame_value_pairs = list(zip(data['frames'], data['mask_size']))
            mask_size_data[track_id] = frame_value_pairs
        
        # Update plots
        self.intersection_pixels_plot.update_plot(self.current_frame, intersection_pixels_data)
        self.engagement_plot.update_plot(self.current_frame, engagement_data)
        self.mask_size_plot.update_plot(self.current_frame, mask_size_data)
    
    def change_episode(self, index):
        """Change current episode"""
        if self.edit_mode:
            # Automatically save changes and exit edit mode
            if self.edited_detections:
                # Save changes silently (no message box)
                try:
                    # Create backup
                    session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
                    annotations_dir = self.get_annotations_dir(session, episode)
                    detection_file = os.path.join(annotations_dir, "annotations_360.json")
                    
                    # Ensure annotations directory exists
                    os.makedirs(annotations_dir, exist_ok=True)
                    
                    if os.path.exists(detection_file):
                        # Create timestamped backup
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_file = os.path.join(annotations_dir, f"annotations_360_backup_{timestamp}.json")
                        import shutil
                        shutil.copy2(detection_file, backup_file)
                    
                    # Apply changes to original detections
                    for frame_idx, frame_data in self.edited_detections.items():
                        if frame_idx in self.detections:
                            # Replace original detections with edited ones
                            self.detections[frame_idx]["detections"] = frame_data["detections"]
                    
                    # Save to file
                    self.save_detections_to_file()
                    
                    # Clear edited detections
                    self.edited_detections = {}
                    
                    self.statusBar().showMessage("Changes saved automatically before switching episode")
                    prInfo("Changes saved automatically before switching episode")
                except Exception as e:
                    prError(f"Failed to save changes automatically: {str(e)}")
                    # Reset combo box to current episode on error
                    self.episode_combo.setCurrentIndex(self.current_episode_idx)
                    return
            
            # Keep edit mode enabled (don't exit edit mode)
            # Edit mode stays on when switching episodes
            
        if 0 <= index < len(self.episodes_list):
            if self.is_playing:
                self.pause_playback()
            
            # Stop preload worker before switching episodes (prevents core dump)
            self.stop_preload_worker()
            
            # Clean up video capture from previous episode
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.video_frames = None
            self.video_path = None
            
            self.current_episode_idx = index
            self.load_episode_data()
            
            # Update next episode button state after loading
            self.update_next_episode_button_state()
    
    def previous_episode(self):
        """Navigate to the previous episode"""
        if self.current_episode_idx > 0:
            self.episode_combo.setCurrentIndex(self.current_episode_idx - 1)
    
    def next_episode(self):
        """Navigate to the next episode"""
        if self.current_episode_idx < len(self.episodes_list) - 1:
            self.episode_combo.setCurrentIndex(self.current_episode_idx + 1)
    
    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start automatic playback"""
        self.is_playing = True
        self.play_pause_btn.setText("⏸ Pause [Space]")
        self.play_timer.start(1)  # 1ms interval
        self.statusBar().showMessage(f"Playing... (skip: {self.play_skip})")
    
    def pause_playback(self):
        """Pause automatic playback"""
        self.is_playing = False
        self.play_pause_btn.setText("▶ Play [Space]")
        self.play_timer.stop()
        self.statusBar().showMessage("Playback paused")
    
    def play_next_frame(self):
        """Advance to the next frame during playback"""
        if not self.is_playing:
            return
        
        # Calculate effective skip based on auto speed adjustment
        effective_skip = self.play_skip
        
        if self.auto_speed_adjustment:
            # Check next 45 frames for big masks or engaged tracks
            has_activity = self.check_frames_for_activity(self.current_frame, num_frames=45)
            
            if not has_activity:
                # No big masks or engaged tracks, speed up by 5x
                effective_skip = self.play_skip * 5
            # Otherwise use normal play_skip
        
        next_frame = self.current_frame + effective_skip
        
        if next_frame >= len(self.image_files):
            self.pause_playback()
            self.statusBar().showMessage("Playback finished")
            return
        
        self.frame_slider.setValue(next_frame)
    
    def update_skip(self, value):
        """Update skip parameter"""
        self.play_skip = value
    
    def toggle_auto_speed_adjustment(self, checked):
        """Toggle automatic speed adjustment"""
        self.auto_speed_adjustment = checked
    
    def check_frames_for_activity(self, start_frame, num_frames=45):
        """Check if there are big masks or engaged tracks in the next frames
        
        Returns:
            bool: True if big masks (>100k pixels) or engaged tracks (level 3) found, False otherwise
        """
        end_frame = min(start_frame + num_frames, len(self.image_files))
        
        for frame_idx in range(start_frame, end_frame):
            # Get detections for this frame (use edited if available)
            if self.edit_mode and frame_idx in self.edited_detections:
                detections = self.edited_detections[frame_idx]["detections"]
            elif frame_idx in self.detections:
                detections = self.detections[frame_idx]["detections"]
            else:
                continue
            
            for det in detections:
                # Check for big mask (>50,000 pixels)
                if "mask_size" in det:
                    mask_size = det.get("mask_size", 0)
                    if mask_size > 50000:
                        return True
                
                # Check for engaged track (engagement_level == 3)
                engagement_level = det.get("engagement_level", det.get("engagement_level_auto", -1))
                if engagement_level == 3:
                    return True
        
        return False
    
    # Display option toggles
    def toggle_show_masks(self, checked):
        self.show_masks = checked
        self.update_image_display()
    
    def toggle_show_boxes(self, checked):
        self.show_boxes = checked
        self.update_image_display()
    
    def toggle_show_track_ids(self, checked):
        self.show_track_ids = checked
        self.update_image_display()
    
    def update_mask_opacity(self, value):
        """Update mask opacity percentage"""
        self.mask_opacity = value
        self.update_image_display()
    
    def toggle_show_interaction_highlights(self, checked):
        self.show_interaction_highlights = checked
        self.update_image_display()
    
    def toggle_show_interaction_zone(self, checked):
        self.show_interaction_zone = checked
        self.update_image_display()
    
    def toggle_enable_image_rolling(self, checked):
        """Toggle image rolling to center interaction mask"""
        self.enable_image_rolling = checked
        self.update_image_display()
    
    def toggle_show_vitpose_poses(self, checked):
        self.show_vitpose_poses = checked
        self.update_image_display()
    
    def toggle_show_sapiens_poses(self, checked):
        self.show_sapiens_poses = checked
        self.update_image_display()
    
    def update_vitpose_min_score(self, value):
        """Update ViTPose minimum score threshold"""
        self.vitpose_min_score = value / 100.0
        if self.show_vitpose_poses:
            self.update_image_display()
    
    def update_sapiens_min_score(self, value):
        """Update Sapiens minimum score threshold"""
        self.sapiens_min_score = value / 100.0
        if self.show_sapiens_poses:
            self.update_image_display()
    
    def toggle_show_invalid_tracks(self, checked):
        """Toggle display of invalid tracks"""
        self.show_invalid_tracks = checked
        self.update_image_display()
        self.update_track_list()
    
    def toggle_edit_mode(self, checked):
        """Toggle edit mode"""
        self.edit_mode = checked
        self.track_edit_widget.setVisible(checked)
        
        if checked:
            # Initialize edit mode with existing detections
            self.edited_detections = {}
            self.initialize_edited_detections()
            self.calculate_next_available_id()
            self.statusBar().showMessage("Edit mode enabled - Episode switching disabled")
        else:
            # Exit edit mode
            if self.edited_detections:
                reply = QMessageBox.question(self, "Unsaved Changes", 
                                           "You have unsaved changes. Do you want to save them before exiting edit mode?",
                                           QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Save:
                    self.save_changes()
                elif reply == QMessageBox.StandardButton.Cancel:
                    self.edit_mode_cb.setChecked(True)
                    return
                else:
                    self.discard_changes()
            
            self.statusBar().showMessage("Edit mode disabled")
        
        self.update_track_list()
    
    def calculate_next_available_id(self):
        """Calculate the next available track ID for splitting"""
        max_id = -1
        for frame_idx, frame_data in self.detections.items():
            for det in frame_data.get("detections", []):
                track_id = det.get("track_id", -1)
                max_id = max(max_id, track_id)
        self.next_available_id = max_id + 1
    
    def initialize_edited_detections(self):
        """Initialize edited_detections with current detections"""
        for frame_idx, frame_data in self.detections.items():
            if "detections" in frame_data:
                # Create a deep copy of the detections for this frame
                self.edited_detections[frame_idx] = {
                    "detections": [det.copy() for det in frame_data["detections"]]
                }
                # Copy other frame data if present
                for key, value in frame_data.items():
                    if key != "detections":
                        self.edited_detections[frame_idx][key] = value
    
    def set_track_validity_valid(self):
        """Set selected track validity to valid"""
        if not self.edit_mode:
            return
        
        selected_tracks = self.get_selected_tracks()
        if not selected_tracks:
            QMessageBox.warning(self, "No Selection", "Please select a track first.")
            return
        
        for track_id in selected_tracks:
            self.modify_track_validity(track_id, "valid")
        
        self.update_image_display()
        self.update_track_list()
    
    def set_track_validity_invalid(self):
        """Set selected track validity to invalid"""
        if not self.edit_mode:
            prWarning("Not in edit mode")
            return
        
        selected_tracks = self.get_selected_tracks()
        if not selected_tracks:
            QMessageBox.warning(self, "No Selection", "Please select a track first.")
            return
        
        for track_id in selected_tracks:
            self.modify_track_validity(track_id, "invalid")
        
        self.update_image_display()
        self.update_track_list()
    
    def set_track_validity_custom(self):
        """Set selected track validity to custom flag"""
        if not self.edit_mode:
            return
        
        selected_tracks = self.get_selected_tracks()
        if not selected_tracks:
            QMessageBox.warning(self, "No Selection", "Please select a track first.")
            return
        
        # Get custom flag from user
        text, ok = QInputDialog.getText(
            self, 
            'Custom Validity Flag',
            f'Enter custom validity flag for {len(selected_tracks)} track(s):',
            text=''
        )
        
        if ok and text:
            for track_id in selected_tracks:
                self.modify_track_validity(track_id, text.strip())
            
            self.update_image_display()
            self.update_track_list()
    
    def modify_track_validity(self, track_id, new_validity):
        """Modify track validity in edited detections"""
        for frame_idx, frame_data in self.edited_detections.items():
            for det in frame_data.get("detections", []):
                if det.get("track_id", -1) == track_id:
                    # Directly modify the detection in edited_detections
                    det["validity"] = new_validity
    
    def split_track(self):
        """Split selected track at current frame"""
        if not self.edit_mode:
            return
        
        selected_tracks = self.get_selected_tracks()
        if len(selected_tracks) != 1:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly one track to split.")
            return
        
        track_id = selected_tracks[0]
        new_track_id = self.next_available_id
        self.next_available_id += 1
        
        # Split track from current frame onwards
        for frame_idx in range(self.current_frame, len(self.edited_detections)):
            if frame_idx in self.edited_detections:
                for det in self.edited_detections[frame_idx].get("detections", []):
                    if det.get("track_id", -1) == track_id:
                        # Directly modify the track ID in edited_detections
                        det["track_id"] = new_track_id
        
        self.statusBar().showMessage(f"Track {track_id} split at frame {self.current_frame}. New track ID: {new_track_id}")
        self.update_image_display()
        self.update_track_list()
    
    def merge_tracks(self):
        """Merge selected tracks"""
        if not self.edit_mode:
            return
        
        selected_tracks = self.get_selected_tracks()
        if len(selected_tracks) != 2:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly two tracks to merge.")
            return
        
        new_track_id = self.next_available_id
        self.next_available_id += 1
        
        for frame_idx, frame_data in self.edited_detections.items():
            det_tracks_ids_in_frame = [det["track_id"] for det in frame_data["detections"]]
            det_idxs_tracks_ids_to_merge = [idx for idx, track_id in enumerate(det_tracks_ids_in_frame) if track_id in selected_tracks]
            if len(det_idxs_tracks_ids_to_merge) == 1:
                frame_data["detections"][det_idxs_tracks_ids_to_merge[0]]["track_id"] = new_track_id
                
            elif len(det_idxs_tracks_ids_to_merge) == 2:
                # merge the two tracks into one
                det1 = frame_data["detections"][det_idxs_tracks_ids_to_merge[0]]
                det1_mask_size = det1["mask_size"]
                det2 = frame_data["detections"][det_idxs_tracks_ids_to_merge[1]]
                det2_mask_size = det2["mask_size"]
                
                biggest_det = det1 if det1_mask_size > det2_mask_size else det2
                smallest_det = det2 if det1_mask_size > det2_mask_size else det1
                
                # Use full image dimensions for mask decoding
                full_height = self.full_image_height if self.use_mini else self.image_height
                full_width = self.full_image_width if self.use_mini else self.image_width
                biggest_det_mask = decode_RLE(torch.tensor([int(x) for x in biggest_det["mask_rle"].split(",")] , dtype=torch.int32).to(MASK_RLE_DEVICE), (full_height, full_width))
                smallest_det_mask = decode_RLE(torch.tensor([int(x) for x in smallest_det["mask_rle"].split(",")], dtype=torch.int32).to(MASK_RLE_DEVICE), (full_height, full_width))
                
                merged_mask = biggest_det_mask | smallest_det_mask
                merged_mask_rle = encode_RLE(merged_mask)
                merged_mask_size = merged_mask.sum().item()
                merged_mask_rle_str = ",".join([str(x) for x in merged_mask_rle.tolist()])
                
                merged_det = biggest_det.copy()
                merged_det["track_id"] = new_track_id
                merged_det["mask_rle"] = merged_mask_rle_str
                merged_det["mask_size"] = merged_mask_size
                
                engagemement_level_biggest_det = biggest_det["engagement_level"] if "engagement_level" in biggest_det else 0
                engagemement_level_smallest_det = smallest_det["engagement_level"] if "engagement_level" in smallest_det else 0
                engagemement_level_auto_biggest_det = biggest_det["engagement_level_auto"] if "engagement_level_auto" in biggest_det else 0
                engagemement_level_auto_smallest_det = smallest_det["engagement_level_auto"] if "engagement_level_auto" in smallest_det else 0
                
                merged_det["engagement_level"] = max(engagemement_level_biggest_det, engagemement_level_smallest_det)
                merged_det["engagement_level_auto"] = max(engagemement_level_auto_biggest_det, engagemement_level_auto_smallest_det)
                merged_det["xmin"] = min(biggest_det["xmin"], smallest_det["xmin"])
                merged_det["ymin"] = min(biggest_det["ymin"], smallest_det["ymin"])
                merged_det["xmax"] = max(biggest_det["xmax"], smallest_det["xmax"])
                merged_det["ymax"] = max(biggest_det["ymax"], smallest_det["ymax"])
                
                frame_data["detections"].append(merged_det)
                frame_data["detections"] = [det for det in frame_data["detections"] if det["track_id"] not in selected_tracks] # remove older tracks
                
            elif len(det_idxs_tracks_ids_to_merge) > 2:
                # should not happen
                raise ValueError(f"More than 2 tracks to merge in frame {frame_idx}: {det_idxs_tracks_ids_to_merge}")
            else:
                # do nothing we have 0 tracks to merge
                pass
        
        # Merge engagement data from both tracks
        self.merge_track_engagement_data(selected_tracks, new_track_id)
        
        self.statusBar().showMessage(f"Merged tracks {selected_tracks} into track {new_track_id}")
        self.update_image_display()
        self.update_track_list()
        self.update_plots()
    
    def merge_track_engagement_data(self, old_track_ids, new_track_id):
        """Merge engagement data from multiple tracks into a new merged track"""
        # Collect all engagement data from the tracks to merge
        merged_frames_eng = {}  # frame -> engagement_level
        
        for track_id in old_track_ids:
            if track_id in self.track_engagement_data:
                frames = self.track_engagement_data[track_id]['frames']
                engagement = self.track_engagement_data[track_id]['engagement']
                
                for frame, eng_level in zip(frames, engagement):
                    # If frame already exists, take the maximum engagement level
                    if frame in merged_frames_eng:
                        merged_frames_eng[frame] = max(merged_frames_eng[frame], eng_level)
                    else:
                        merged_frames_eng[frame] = eng_level
        
        # Create new engagement data for the merged track
        if merged_frames_eng:
            # Sort by frame number
            sorted_frames = sorted(merged_frames_eng.keys())
            sorted_engagement = [merged_frames_eng[frame] for frame in sorted_frames]
            
            self.track_engagement_data[new_track_id] = {
                'frames': sorted_frames,
                'engagement': sorted_engagement
            }
        
        # Also merge mask size data
        merged_frames_mask = {}  # frame -> mask_size
        
        for track_id in old_track_ids:
            if track_id in self.track_mask_size_data:
                frames = self.track_mask_size_data[track_id]['frames']
                mask_sizes = self.track_mask_size_data[track_id]['mask_size']
                
                for frame, mask_size in zip(frames, mask_sizes):
                    # If frame already exists, take the maximum mask size
                    if frame in merged_frames_mask:
                        merged_frames_mask[frame] = max(merged_frames_mask[frame], mask_size)
                    else:
                        merged_frames_mask[frame] = mask_size
        
        if merged_frames_mask:
            sorted_frames_mask = sorted(merged_frames_mask.keys())
            sorted_mask_sizes = [merged_frames_mask[frame] for frame in sorted_frames_mask]
            
            self.track_mask_size_data[new_track_id] = {
                'frames': sorted_frames_mask,
                'mask_size': sorted_mask_sizes
            }
        
        # Also merge intersection pixels data
        merged_frames_intersection = {}  # frame -> intersection_pixels
        
        for track_id in old_track_ids:
            if track_id in self.track_intersection_pixels_data:
                frames = self.track_intersection_pixels_data[track_id]['frames']
                intersection_pixels = self.track_intersection_pixels_data[track_id]['intersection_pixels']
                
                for frame, pixels in zip(frames, intersection_pixels):
                    # If frame already exists, take the maximum intersection pixels
                    if frame in merged_frames_intersection:
                        merged_frames_intersection[frame] = max(merged_frames_intersection[frame], pixels)
                    else:
                        merged_frames_intersection[frame] = pixels
        
        if merged_frames_intersection:
            sorted_frames_intersection = sorted(merged_frames_intersection.keys())
            sorted_intersection_pixels = [merged_frames_intersection[frame] for frame in sorted_frames_intersection]
            
            self.track_intersection_pixels_data[new_track_id] = {
                'frames': sorted_frames_intersection,
                'intersection_pixels': sorted_intersection_pixels
            }
        
        # Remove old track data
        for track_id in old_track_ids:
            if track_id in self.track_engagement_data:
                del self.track_engagement_data[track_id]
            if track_id in self.track_mask_size_data:
                del self.track_mask_size_data[track_id]
            if track_id in self.track_intersection_pixels_data:
                del self.track_intersection_pixels_data[track_id]
    
    def handle_e_key(self):
        """Handle E key press: engagement correction if in edit mode with track selected"""
        if self.edit_mode:
            # Check if exactly one track is selected
            selected_tracks = self.get_selected_tracks()
            if len(selected_tracks) == 1:
                # Open engagement correction dialog
                self.open_engagement_correction_dialog()
            else:
                # No track or multiple tracks selected
                QMessageBox.warning(self, "Invalid Selection", "Please select exactly one track to correct engagement level.")
    
    def set_engagement_to_zero(self):
        """Set selected track's engagement to 0 for ±15 frames"""
        if not self.edit_mode:
            QMessageBox.warning(self, "Edit Mode Required", "Please enable edit mode first.")
            return
        
        # Check that exactly one track is selected
        selected_tracks = self.get_selected_tracks()
        if len(selected_tracks) != 1:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly one track.")
            return
        
        track_id = selected_tracks[0]
        
        # Apply engagement correction: set to 0 for ±15 frames
        self.apply_engagement_correction(track_id, 0, 15, 15)
    
    def set_engagement_to_three(self):
        """Set selected track's engagement to 3 for the next 15 frames"""
        if not self.edit_mode:
            QMessageBox.warning(self, "Edit Mode Required", "Please enable edit mode first.")
            return
        
        # Check that exactly one track is selected
        selected_tracks = self.get_selected_tracks()
        if len(selected_tracks) != 1:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly one track.")
            return
        
        track_id = selected_tracks[0]
        
        # Apply engagement correction: set to 3 for next 15 frames only (0 past frames)
        self.apply_engagement_correction(track_id, 3, 0, 15)
    
    def open_engagement_correction_dialog(self):
        """Open dialog for correcting engagement levels"""
        if not self.edit_mode:
            QMessageBox.warning(self, "Edit Mode Required", "Please enable edit mode first.")
            return
        
        # Check that exactly one track is selected
        selected_tracks = self.get_selected_tracks()
        if len(selected_tracks) != 1:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly one track to correct engagement level.")
            return
        
        track_id = selected_tracks[0]
        
        # Get engagement data for this track
        engagement_data = self.track_engagement_data.get(track_id, {'frames': [], 'engagement': []})
        
        # Create and show dialog
        dialog = EngagementCorrectionDialog(
            self,
            track_id,
            self.current_frame,
            len(self.image_files),
            engagement_data
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.accepted_correction:
            # Apply the correction
            self.apply_engagement_correction(
                track_id,
                dialog.new_engagement,
                dialog.past_frames,
                dialog.future_frames
            )
    
    def apply_engagement_correction(self, track_id, new_engagement, past_frames, future_frames):
        """Apply engagement level correction to the specified track and frame range"""
        # Calculate frame range
        frame_start = max(0, self.current_frame - past_frames)
        frame_end = min(len(self.image_files) - 1, self.current_frame + future_frames)
        
        # Initialize edited detections if needed
        if not self.edited_detections:
            self.initialize_edited_detections()
        
        # Apply correction to all frames in range
        frames_modified = 0
        for frame_idx in range(frame_start, frame_end + 1):
            if frame_idx in self.edited_detections:
                for det in self.edited_detections[frame_idx].get("detections", []):
                    if det.get("track_id", -1) == track_id:
                        # Update engagement level
                        det["engagement_level"] = new_engagement
                        # Also update engagement_level_auto if it exists
                        if "engagement_level_auto" in det:
                            det["engagement_level_auto"] = new_engagement
                        frames_modified += 1
        
        # Update engagement plot data
        # Initialize if needed
        if track_id not in self.track_engagement_data:
            self.track_engagement_data[track_id] = {'frames': [], 'engagement': []}
        
        # Update the engagement data for this track
        orig_frames = self.track_engagement_data[track_id]['frames']
        orig_engagement = self.track_engagement_data[track_id]['engagement']
        
        # Create a set of frames that already have engagement data
        existing_frames = set(orig_frames)
        
        # Create updated lists
        updated_frames = []
        updated_engagement = []
        
        # Process existing frames
        for frame, engagement in zip(orig_frames, orig_engagement):
            if frame_start <= frame <= frame_end:
                # Use new engagement level
                updated_frames.append(frame)
                updated_engagement.append(new_engagement)
            else:
                # Keep original
                updated_frames.append(frame)
                updated_engagement.append(engagement)
        
        # Add engagement data for frames in range that don't have it yet but have the track
        for frame_idx in range(frame_start, frame_end + 1):
            if frame_idx not in existing_frames:
                # Check if this frame has the track
                if frame_idx in self.edited_detections:
                    for det in self.edited_detections[frame_idx].get("detections", []):
                        if det.get("track_id", -1) == track_id:
                            # Add new engagement data
                            updated_frames.append(frame_idx)
                            updated_engagement.append(new_engagement)
                            break
        
        # Sort by frame number
        sorted_pairs = sorted(zip(updated_frames, updated_engagement))
        updated_frames = [f for f, e in sorted_pairs]
        updated_engagement = [e for f, e in sorted_pairs]
        
        # Update the data
        self.track_engagement_data[track_id]['frames'] = updated_frames
        self.track_engagement_data[track_id]['engagement'] = updated_engagement
        
        self.statusBar().showMessage(f"Applied engagement correction to track {track_id}: {new_engagement} for frames {frame_start}-{frame_end} ({frames_modified} detections modified)")
        prInfo(f"Engagement correction applied: Track {track_id}, Level {new_engagement}, Frames {frame_start}-{frame_end}")
        
        # Update display
        self.update_image_display()
        self.update_plots()
    
    def open_all_tracks_engagement_dialog(self):
        """Open dialog to set engagement for all tracks in a frame range"""
        if not self.edit_mode:
            QMessageBox.warning(self, "Edit Mode Required", "Please enable edit mode first.")
            return
        
        # Open dialog
        dialog = AllTracksEngagementDialog(
            self,
            self.current_frame,
            len(self.image_files)
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.accepted:
            # Show warning before applying
            frame_start = max(0, self.current_frame - dialog.past_frames)
            frame_end = min(len(self.image_files) - 1, self.current_frame + dialog.future_frames)
            num_frames = frame_end - frame_start + 1
            
            reply = QMessageBox.warning(
                self,
                "Confirm Engagement Update",
                f"This will set engagement level {dialog.new_engagement} for ALL tracks "
                f"in frames {frame_start} to {frame_end} ({num_frames} frames).\n\n"
                f"This action cannot be undone easily. Are you sure you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.apply_all_tracks_engagement(
                    dialog.new_engagement,
                    dialog.past_frames,
                    dialog.future_frames
                )
    
    def apply_all_tracks_engagement(self, new_engagement, past_frames, future_frames):
        """Apply engagement level to all tracks in the specified frame range"""
        # Calculate frame range
        frame_start = max(0, self.current_frame - past_frames)
        frame_end = min(len(self.image_files) - 1, self.current_frame + future_frames)
        
        # Initialize edited detections if needed
        if not self.edited_detections:
            self.initialize_edited_detections()
        
        # Track all affected track IDs
        affected_tracks = set()
        total_detections_modified = 0
        
        # Apply correction to all tracks in all frames in range
        for frame_idx in range(frame_start, frame_end + 1):
            if frame_idx in self.edited_detections:
                for det in self.edited_detections[frame_idx].get("detections", []):
                    track_id = det.get("track_id", -1)
                    if track_id >= 0:  # Valid track ID
                        # Update engagement level
                        det["engagement_level"] = new_engagement
                        # Also update engagement_level_auto if it exists
                        if "engagement_level_auto" in det:
                            det["engagement_level_auto"] = new_engagement
                        affected_tracks.add(track_id)
                        total_detections_modified += 1
        
        # Update engagement plot data for all affected tracks
        for track_id in affected_tracks:
            # Initialize if needed
            if track_id not in self.track_engagement_data:
                self.track_engagement_data[track_id] = {'frames': [], 'engagement': []}
            
            # Update the engagement data for this track
            orig_frames = self.track_engagement_data[track_id]['frames']
            orig_engagement = self.track_engagement_data[track_id]['engagement']
            
            # Create a set of frames that already have engagement data
            existing_frames = set(orig_frames)
            
            # Create updated lists
            updated_frames = []
            updated_engagement = []
            
            # Process existing frames
            for frame, engagement in zip(orig_frames, orig_engagement):
                if frame_start <= frame <= frame_end:
                    # Use new engagement level
                    updated_frames.append(frame)
                    updated_engagement.append(new_engagement)
                else:
                    # Keep original
                    updated_frames.append(frame)
                    updated_engagement.append(engagement)
            
            # Add engagement data for frames in range that don't have it yet but have the track
            for frame_idx in range(frame_start, frame_end + 1):
                if frame_idx not in existing_frames:
                    # Check if this frame has the track
                    if frame_idx in self.edited_detections:
                        for det in self.edited_detections[frame_idx].get("detections", []):
                            if det.get("track_id", -1) == track_id:
                                # Add new engagement data
                                updated_frames.append(frame_idx)
                                updated_engagement.append(new_engagement)
                                break
            
            # Sort by frame number
            sorted_pairs = sorted(zip(updated_frames, updated_engagement))
            updated_frames = [f for f, e in sorted_pairs]
            updated_engagement = [e for f, e in sorted_pairs]
            
            # Update the data
            self.track_engagement_data[track_id]['frames'] = updated_frames
            self.track_engagement_data[track_id]['engagement'] = updated_engagement
        
        self.statusBar().showMessage(
            f"Applied engagement {new_engagement} to {len(affected_tracks)} tracks "
            f"in frames {frame_start}-{frame_end} ({total_detections_modified} detections modified)"
        )
        prInfo(
            f"Applied engagement {new_engagement} to all tracks: "
            f"{len(affected_tracks)} tracks, Frames {frame_start}-{frame_end}, "
            f"{total_detections_modified} detections modified"
        )
        
        # Update display
        self.update_image_display()
        self.update_plots()
    
    def is_track_selected(self, track_id):
        """Check if a track is selected (either single selection or in list selection)"""
        # Check single selection
        if self.selected_track is not None and self.selected_track == track_id:
            return True
        # Check list selection
        for item in self.track_list.selectedItems():
            item_track_id = item.data(Qt.ItemDataRole.UserRole)
            if item_track_id == track_id:
                return True
        return False
    
    def get_selected_tracks(self):
        """Get list of selected track IDs"""
        selected_tracks = []
        # Include single selection if set
        if self.selected_track is not None:
            selected_tracks.append(self.selected_track)
        # Add tracks from list selection (avoid duplicates)
        for item in self.track_list.selectedItems():
            track_id = item.data(Qt.ItemDataRole.UserRole)
            if track_id is not None and track_id not in selected_tracks:
                selected_tracks.append(track_id)
        return selected_tracks
    
    def save_changes(self):
        """Save edited detections to file with backup"""
        if not self.edited_detections:
            QMessageBox.information(self, "No Changes", "No changes to save.")
            return
        
        try:
            # Create backup
            session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
            annotations_dir = self.get_annotations_dir(session, episode)
            detection_file = os.path.join(annotations_dir, "annotations_360.json")
            
            # Ensure annotations directory exists
            os.makedirs(annotations_dir, exist_ok=True)
            
            if os.path.exists(detection_file):
                # Create timestamped backup
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(annotations_dir, f"annotations_360_backup_{timestamp}.json")
                import shutil
                shutil.copy2(detection_file, backup_file)
                self.statusBar().showMessage(f"Backup created: {os.path.basename(backup_file)}")
            
            # Apply changes to original detections
            for frame_idx, frame_data in self.edited_detections.items():
                if frame_idx in self.detections:
                    # Replace original detections with edited ones
                    self.detections[frame_idx]["detections"] = frame_data["detections"]
            
            # Save to file
            self.save_detections_to_file()
            
            # Clear edited detections
            self.edited_detections = {}
            
            QMessageBox.information(self, "Success", "Changes saved successfully!")
            self.statusBar().showMessage("Changes saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save changes: {str(e)}")
    
    def discard_changes(self):
        """Discard all edited changes"""
        self.edited_detections = {}
        self.update_image_display()
        self.update_track_list()
        self.statusBar().showMessage("Changes discarded")
    
    def update_episode_flag(self, new_flag):
        """Update the episode flag and save to file"""
        if self.current_episodes_flags_path is None:
            QMessageBox.warning(self, "Error", "No episodes flags path available.")
            return False
        
        try:
            # Get current episode info
            session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
            
            # Update the flag in the dictionary
            if episode in associated_episodes_flags_dict:
                associated_episodes_flags_dict[episode]["flag"] = new_flag
            else:
                QMessageBox.warning(self, "Error", f"Episode {episode} not found in flags dictionary.")
                return False
            
            # Save to file
            write_dic_to_json_file(associated_episodes_flags_dict, self.current_episodes_flags_path)
            
            # Update the episode combo box text
            episode_flag = associated_episodes_flags_dict[episode]["flag"]
            item_text = f"{session} | Ep. {episode} | {episode_flag}"
            self.episode_combo.setItemText(self.current_episode_idx, item_text)
            
            self.statusBar().showMessage(f"Episode flag updated to: {new_flag}")
            prInfo(f"Updated episode {session}/{episode} flag to: {new_flag}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update episode flag: {str(e)}")
            return False
    
    def append_checked_flag(self):
        """Append ',checked' to the current episode flag"""
        session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
        current_flag = associated_episodes_flags_dict[episode]["flag"]
        
        # Append ",checked" if not already present
        if ",checked" not in current_flag:
            new_flag = current_flag + ",checked"
            self.update_episode_flag(new_flag)
        else:
            self.statusBar().showMessage("Flag already contains ',checked'")
    
    def set_invalid_flag(self):
        """Set episode flag to 'invalid'"""
        self.update_episode_flag("invalid")
    
    def set_valid_flag(self):
        """Set episode flag to 'valid'"""
        self.update_episode_flag("valid")
    
    def save_detections_to_file(self):
        """Save current detections to JSON file"""
        session, episode, associated_episodes_flags_dict = self.episodes_list[self.current_episode_idx]
        annotations_dir = self.get_annotations_dir(session, episode)
        detection_file = os.path.join(annotations_dir, "annotations_360.json")
        
        # Ensure annotations directory exists
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Convert detections back to list format
        detections_list = []
        for frame_idx in sorted(self.detections.keys()):
            frame_data = self.detections[frame_idx]
            if "image_file" in frame_data:
                detections_list.append(frame_data)
        
        # Write to file
        with open(detection_file, 'w') as f:
            json.dump(detections_list, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Interact360 Visualizer")
    parser.add_argument("--directory", "-d", type=str, default="./demo/output/",
                       help="Output directory containing processed data (e.g., demo/output), default: %(default)s")
    parser.add_argument("--recording_name", "-rn", type=str, default="ClubInteract-TCAn",
                       help="Recording name to visualize, if not provided will visualize all recordings in the directory (default: %(default)s)")
    parser.add_argument("--miniatures", "-mini", action="store_true", default=False,
                       help="Use miniature images (4x smaller) from images_360_miniatures_scale_4 (default: %(default)s)")
    parser.add_argument("--splitted_dataset", "-sd", action="store_true", default=False,
                       help="Use splitted dataset mode, the data and the annotations are separated into different directories (default: %(default)s)")
    parser.add_argument("--video_mode", "-vm", action="store_true", default=False,
                       help="Use video mode, load videos (full or mini) instead of images for display (default: %(default)s)")
    parser.add_argument("--select_tracks_from_report", type=str, default=None,
                       help="Path to report YAML file to filter tracks based on prediction results")
    parser.add_argument("--select_tracks_based_on_prediction", type=str, default="all",
                       choices=["tp", "fp", "tn", "fn", "all"],
                       help="Filter tracks based on prediction status: tp (true positive), fp (false positive), tn (true negative), fn (false negative), or all (default: %(default)s)")
    parser.add_argument("--prediction_threshold", type=float, default=None,
                       help="Custom threshold for predicted_score to determine predicted_label. If provided, predicted_label = 1 if predicted_score >= threshold, else 0. If not provided, uses predicted_label from report (default: %(default)s)")
    args = parser.parse_args()

    # Validate arguments
    if not os.path.isdir(args.directory):
        prError(f"Output directory does not exist: {args.directory}")
        sys.exit(1)
    
    # Create and run application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = Interact360Visualizer(args)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
