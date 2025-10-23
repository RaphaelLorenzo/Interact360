#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import numpy as np
import cv2
import json
from typing import List, Dict, Optional, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from misc.print_utils import *
from misc.other_utils import *
from misc.visualize_utils import GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS, COCO_EDGES

# Import torch and RLE tools
import torch
from misc.rle_tools import decode_RLE

MASK_RLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSlider, QPushButton, QCheckBox, QListWidget, QLabel, QSplitter,
        QGroupBox, QGridLayout, QListWidgetItem, QMessageBox, QStatusBar,
        QSpinBox, QTabWidget, QComboBox
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeySequence, QShortcut
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
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 400)
        self.setScaledContents(False)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_factor = (1.0, 1.0)
        self.image_size = (0, 0)
        self.original_pixmap = None
        
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


class Interact360Visualizer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Find episodes
        self.episodes_list = self.find_episodes()
        self.current_episode_idx = 0
        
        # Data storage
        self.image_files = []
        self.detections = {}
        self.current_frame = 0
        
        # Track data for plots
        self.track_engagement_data = {}  # track_id -> {'frames': [], 'engagement': []}
        self.track_mask_size_data = {}  # track_id -> {'frames': [], 'mask_size': []}
        
        # Interaction mask
        self.interaction_mask = None
        self.interact_zone_tracking = None  # Dynamic interaction zones per frame
        self.load_interaction_mask()
        
        # Display options
        self.show_masks = False
        self.show_boxes = True
        self.show_track_ids = True
        self.show_interaction_highlights = True
        self.show_interaction_zone = False  # Option to constantly display interaction zone
        self.mask_opacity = 30  # Default 30% opacity
        
        # Pose visualization options
        self.show_vitpose_poses = False
        self.show_sapiens_poses = False
        self.vitpose_min_score = 0.6  # Minimum score for ViTPose keypoints
        self.sapiens_min_score = 0.6  # Minimum score for Sapiens keypoints
        
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
        
        self.init_ui()
        self.load_episode_data()
    
    def find_episodes(self):
        """Find available episodes in the output directory"""
        episodes_list = []
        
        if not os.path.exists(self.args.directory):
            prError(f"Output directory does not exist: {self.args.directory}")
            return episodes_list
        
    
        if self.args.recording_name != "" and not os.path.exists(os.path.join(self.args.directory, self.args.recording_name)):
            prError(f"Recording name does not exist: {self.args.recording_name} in directory {self.args.directory}")
            return episodes_list
        
        if self.args.recording_name == "":
            # Look for subdirectories that contain episodes
            for item in os.listdir(self.args.directory):
                item_path = os.path.join(self.args.directory, item)
                if os.path.isdir(item_path):
                    episodes_dir = os.path.join(item_path, "episodes")
                    if os.path.exists(episodes_dir):
                        # Find episode subdirectories (like 0000)
                        for ep in os.listdir(episodes_dir):
                            ep_path = os.path.join(episodes_dir, ep)
                            if os.path.isdir(ep_path) and ep.isdigit():
                                episodes_list.append((item, ep))
        else:
            episodes_dir = os.path.join(self.args.directory, self.args.recording_name, "episodes")
            if os.path.exists(episodes_dir):
                # Find episode subdirectories (like 0000)
                for ep in os.listdir(episodes_dir):
                    ep_path = os.path.join(episodes_dir, ep)
                    if os.path.isdir(ep_path) and ep.isdigit():
                        episodes_list.append((self.args.recording_name, ep))
                        
        episodes_list.sort()
        prInfo(f"Found {len(episodes_list)} episodes: {episodes_list}")
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
        interact_zone_tracking_path = os.path.join(episode_dir, "interact_zone_tracking_360.json")
        
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
        control_panel.setMaximumWidth(300)
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
        main_splitter.setSizes([250, 800, 350])
        
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
        for i, (session, ep) in enumerate(self.episodes_list):
            item_text = f"{session} | Ep. {ep}"
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
        play_layout = QVBoxLayout(play_group)
        
        self.play_pause_btn = QPushButton("▶ Play [Space]")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        play_layout.addWidget(self.play_pause_btn)
        
        # Skip parameter
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Skip:"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 20)
        self.skip_spin.setValue(self.play_skip)
        self.skip_spin.valueChanged.connect(self.update_skip)
        skip_layout.addWidget(self.skip_spin)
        play_layout.addLayout(skip_layout)
        
        layout.addWidget(play_group)
        
        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)
        
        self.show_masks_cb = QCheckBox("Show Masks [M]")
        self.show_masks_cb.setChecked(False)
        self.show_masks_cb.toggled.connect(self.toggle_show_masks)
        options_layout.addWidget(self.show_masks_cb)
        
        self.show_boxes_cb = QCheckBox("Show Bounding Boxes [B]")
        self.show_boxes_cb.setChecked(True)
        self.show_boxes_cb.toggled.connect(self.toggle_show_boxes)
        options_layout.addWidget(self.show_boxes_cb)
        
        self.show_track_ids_cb = QCheckBox("Show Track IDs [T]")
        self.show_track_ids_cb.setChecked(True)
        self.show_track_ids_cb.toggled.connect(self.toggle_show_track_ids)
        options_layout.addWidget(self.show_track_ids_cb)
        
        self.show_interaction_highlights_cb = QCheckBox("Show Interaction Highlights [I]")
        self.show_interaction_highlights_cb.setChecked(True)
        self.show_interaction_highlights_cb.setEnabled(self.interaction_mask is not None)
        self.show_interaction_highlights_cb.toggled.connect(self.toggle_show_interaction_highlights)
        options_layout.addWidget(self.show_interaction_highlights_cb)
        
        self.show_interaction_zone_cb = QCheckBox("Show Interaction Zone [Z]")
        self.show_interaction_zone_cb.setChecked(False)
        self.show_interaction_zone_cb.setEnabled(self.interaction_mask is not None)
        self.show_interaction_zone_cb.toggled.connect(self.toggle_show_interaction_zone)
        options_layout.addWidget(self.show_interaction_zone_cb)
        
        # Pose visualization options
        self.show_vitpose_poses_cb = QCheckBox("Show ViTPose Poses [P]")
        self.show_vitpose_poses_cb.setChecked(False)
        self.show_vitpose_poses_cb.toggled.connect(self.toggle_show_vitpose_poses)
        options_layout.addWidget(self.show_vitpose_poses_cb)
        
        self.show_sapiens_poses_cb = QCheckBox("Show Sapiens Poses [Shift+P]")
        self.show_sapiens_poses_cb.setChecked(False)
        self.show_sapiens_poses_cb.toggled.connect(self.toggle_show_sapiens_poses)
        options_layout.addWidget(self.show_sapiens_poses_cb)
        
        # ViTPose minimum score control
        vitpose_score_layout = QHBoxLayout()
        vitpose_score_layout.addWidget(QLabel("ViTPose Min Score:"))
        self.vitpose_score_spin = QSpinBox()
        self.vitpose_score_spin.setRange(0, 100)
        self.vitpose_score_spin.setValue(int(self.vitpose_min_score * 100))
        self.vitpose_score_spin.setSuffix("%")
        self.vitpose_score_spin.valueChanged.connect(self.update_vitpose_min_score)
        vitpose_score_layout.addWidget(self.vitpose_score_spin)
        options_layout.addLayout(vitpose_score_layout)
        
        # Sapiens minimum score control
        sapiens_score_layout = QHBoxLayout()
        sapiens_score_layout.addWidget(QLabel("Sapiens Min Score:"))
        self.sapiens_score_spin = QSpinBox()
        self.sapiens_score_spin.setRange(0, 100)
        self.sapiens_score_spin.setValue(int(self.sapiens_min_score * 100))
        self.sapiens_score_spin.setSuffix("%")
        self.sapiens_score_spin.valueChanged.connect(self.update_sapiens_min_score)
        sapiens_score_layout.addWidget(self.sapiens_score_spin)
        options_layout.addLayout(sapiens_score_layout)
        
        # Mask opacity control
        mask_opacity_layout = QHBoxLayout()
        mask_opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.mask_opacity_spin = QSpinBox()
        self.mask_opacity_spin.setRange(0, 100)
        self.mask_opacity_spin.setValue(30)
        self.mask_opacity_spin.setSuffix("%")
        self.mask_opacity_spin.valueChanged.connect(self.update_mask_opacity)
        mask_opacity_layout.addWidget(self.mask_opacity_spin)
        options_layout.addLayout(mask_opacity_layout)
        
        layout.addWidget(options_group)
        
        # Track list
        track_group = QGroupBox("Tracks")
        track_layout = QVBoxLayout(track_group)
        
        self.track_list = QListWidget()
        self.track_list.setMaximumHeight(200)
        self.track_list.itemClicked.connect(self.on_track_selected)
        track_layout.addWidget(self.track_list)
        
        # Clear selection button
        self.clear_selection_btn = QPushButton("Clear Selection [Esc]")
        self.clear_selection_btn.clicked.connect(self.clear_track_selection)
        track_layout.addWidget(self.clear_selection_btn)
        
        layout.addWidget(track_group)
        
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
        self.image_widget = ImageDisplayWidget()
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
        
        # Track selection
        QShortcut(QKeySequence("Escape"), self, self.clear_track_selection)
        
        # Play/Pause
        QShortcut(QKeySequence("Space"), self, self.toggle_play_pause)
        
        # Episode navigation shortcuts
        QShortcut(QKeySequence("Ctrl+B"), self, self.previous_episode)
        QShortcut(QKeySequence("Ctrl+N"), self, self.next_episode)
    
    def load_episode_data(self):
        """Load data for current episode"""
        if not self.episodes_list:
            prWarning("No episodes available")
            return
            
        session, episode = self.episodes_list[self.current_episode_idx]
        episode_dir = os.path.join(self.args.directory, session, "episodes", episode)
        
        prInfo(f"Loading episode: {session}/{episode}")
        
        # Load images
        images_dir = os.path.join(episode_dir, "images_360")
        self.image_files = self.get_image_files(images_dir)
        
        # Load detections
        detection_file = os.path.join(episode_dir, "sam2_detections_360.json")
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
            prWarning(f"No detection file found: {detection_file}")
        
        # Load interact zone tracking if available
        self.load_interact_zone_tracking(episode_dir)
        
        # Update UI
        self.update_slider()
        self.current_frame = 0
        self.update_image_display()
        self.update_track_list()
        self.update_plots()
        
        # Update interaction UI availability
        has_interaction_data = (self.interaction_mask is not None or self.interact_zone_tracking is not None)
        self.show_interaction_highlights_cb.setEnabled(has_interaction_data)
        self.show_interaction_zone_cb.setEnabled(has_interaction_data)
        
        status_msg = f"Loaded episode {session}/{episode} - {len(self.image_files)} frames, {len(self.detections)} detection frames"
        self.statusBar().showMessage(status_msg)
        prInfo(status_msg)

    
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
            
            # if os.path.basename(image_file) != os.path.basename(det_image_file):
            #     print(f"[{i}] Image file: {os.path.basename(image_file)} vs detection: {os.path.basename(det_image_file)}")
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
            interaction_mask_resized = self.get_interaction_mask_for_frame(frame_index, image_shape)
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
                        return mask.cpu().numpy().astype(bool)

            # Fallback to static mask if dynamic fails or not available
            return self.get_static_interaction_mask(image_shape)
        else:
            # Use static interaction mask
            return self.get_static_interaction_mask(image_shape)
    
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
        
        image_file = self.image_files[self.current_frame]
        # image_file_from_dict = self.detections[self.current_frame]["image_file"]
        detections = self.detections[self.current_frame]["detections"] if self.current_frame in self.detections else []
        
        # Load image
        tic = time.time()
        image = cv2.imread(image_file)
        if image is None:
            prError(f"Could not load image: {image_file}")
            return
        tac = time.time()
        # prTimer(f"Load image {image_file}", tic, tac)
        
        # Draw tracks on the image
        tic = time.time()
        image = self.draw_tracks_on_image(image, detections)
        tac = time.time()
        # prTimer(f"Draw tracks on image", tic, tac)
        
        # Convert BGR to RGB for Qt
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image to widget
        self.image_widget.set_image_with_scale(image_rgb)
        
        # Update image info
        info_text = f"Frame {self.current_frame} - {len(detections)} tracks"
        self.info_label.setText(info_text)
        tac_global = time.time()
        # prTimer(f"Update image display", tic_global, tac_global)
    
    def draw_tracks_on_image(self, image, detections):
        """Draw track visualizations on image"""
        image_copy = image.copy()
        image_height, image_width = image.shape[:2]
        
        # Draw interaction zone as background overlay if enabled
        if self.show_interaction_zone:
            interaction_mask = self.get_interaction_mask_for_frame(self.current_frame, (image_height, image_width))
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
            engagement_level = det.get("engagement_level", det.get("engagement_level_auto", -1))
            
            # Get color
            color = self.get_track_color(track_id)
            
            # Draw bounding box if enabled
            if self.show_boxes:
                thickness = 2
                if (self.selected_track is not None and self.selected_track == track_id):
                    thickness = 4  # Thicker for selected track
                    
                xmin, ymin, xmax, ymax = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
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
                font_scale = 0.7
                font_thickness = 2
                
                text = f"ID: {track_id}"
                x1 = int(det["xmin"])
                if x1 > image_copy.shape[1]:
                    x1 = x1 - image_copy.shape[1]
                y1 = int(det["ymin"])
                
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0], y1), color, -1)
                cv2.putText(image_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
            
            # Draw mask if enabled and available
            if self.show_masks and "mask_rle" in det:
                # Decode RLE mask
                mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                mask = decode_RLE(mask_rle, (image_height, image_width))
                mask_np = mask.cpu().numpy().astype(bool)
                
                # Apply colored overlay
                overlay = image_copy.copy()
                overlay[mask_np] = color
                alpha = self.mask_opacity / 100.0
                image_copy = cv2.addWeighted(image_copy, 1.0-alpha, overlay, alpha, 0)

            # Draw interaction highlights for engagement level 3 tracks
            if (self.show_interaction_highlights and 
                engagement_level == 3 and 
                "mask_rle" in det and 
                self.interaction_mask is not None):
                mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                mask_rle = torch.tensor(mask_rle, dtype=torch.int32)
                intersection = self.get_mask_intersection_with_interaction_zone(
                    mask_rle, (image_height, image_width), self.current_frame)
                
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
            if (self.selected_track is not None and self.selected_track == track_id):
                if self.show_masks and "mask_rle" in det:
                    # Use mask contour for selected track
                    mask_rle = [int(x) for x in det["mask_rle"].split(",")]
                    mask_rle = torch.tensor(mask_rle, dtype=torch.int32).to(MASK_RLE_DEVICE)
                    mask = decode_RLE(mask_rle, (image_height, image_width))
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Draw the largest contour with thick line
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(image_copy, [largest_contour], -1, color, 6)
                else:
                    # Fall back to thick bounding box when masks not available
                    x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
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
                        for i in range(keypoints.shape[0]):
                            if i < len(scores_vitpose) and scores_vitpose[i] > self.vitpose_min_score:
                                kp_x, kp_y = int(keypoints[i, 0]), int(keypoints[i, 1])
                                if kp_x > 0 and kp_y > 0:
                                    cv2.circle(image_copy, (kp_x, kp_y), 3, pose_color, -1)
                        
                        # Draw skeleton connections
                        for edge in COCO_EDGES:
                            if (edge[0] < len(scores_vitpose) and edge[1] < len(scores_vitpose) and 
                                scores_vitpose[edge[0]] > self.vitpose_min_score and scores_vitpose[edge[1]] > self.vitpose_min_score):
                                pt1 = (int(keypoints[edge[0], 0]), int(keypoints[edge[0], 1]))
                                pt2 = (int(keypoints[edge[1], 0]), int(keypoints[edge[1], 1]))
                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    cv2.line(image_copy, pt1, pt2, pose_color, 2)
            
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
                        for i in range(sapiens_keypoints.shape[0]):
                            if sapiens_scores[i] > self.sapiens_min_score:
                                kpt_color = GOLIATH_KPTS_COLORS[i]
                                kp_x, kp_y = int(sapiens_keypoints[i, 0]), int(sapiens_keypoints[i, 1])
                                if kp_x > 0 and kp_y > 0:
                                    cv2.circle(image_copy, (kp_x, kp_y), 3, kpt_color, -1)
                                
                        # Draw skeleton connections
                        for edge_idx in GOLIATH_SKELETON_INFO:
                            edge = GOLIATH_SKELETON_INFO[edge_idx]
                            kp1_score = sapiens_scores[edge["link"][0]]
                            kp2_score = sapiens_scores[edge["link"][1]]
                            if kp1_score > self.sapiens_min_score and kp2_score > self.sapiens_min_score:
                                edge_color = edge["color"]
                                kp_x1, kp_y1 = int(sapiens_keypoints[edge["link"][0], 0]), int(sapiens_keypoints[edge["link"][0], 1])
                                kp_x2, kp_y2 = int(sapiens_keypoints[edge["link"][1], 0]), int(sapiens_keypoints[edge["link"][1], 1])
                                if kp_x1 > 0 and kp_y1 > 0 and kp_x2 > 0 and kp_y2 > 0:
                                    cv2.line(image_copy, (kp_x1, kp_y1), (kp_x2, kp_y2), edge_color, 2)
        
        return image_copy
    
    def update_track_list(self):
        """Update the track list widget"""
        self.track_list.clear()
        
        if self.current_frame not in self.detections:
            return
        
        detections = self.detections[self.current_frame]["detections"]
        
        for det in detections:
            track_id = det.get("track_id", -1)
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
    
    def on_track_selected(self, item):
        """Handle track selection from the list"""
        track_id = item.data(Qt.ItemDataRole.UserRole)
        if track_id is not None:
            self.selected_track = track_id
            self.update_image_display()
    
    def clear_track_selection(self):
        """Clear the current track selection"""
        self.selected_track = None
        self.track_list.clearSelection()
        self.update_image_display()
    
    def update_plots(self):
        """Update both engagement and mask size plots"""
        if not hasattr(self, 'engagement_plot') or not hasattr(self, 'mask_size_plot'):
            return
        
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
        self.engagement_plot.update_plot(self.current_frame, engagement_data)
        self.mask_size_plot.update_plot(self.current_frame, mask_size_data)
    
    def change_episode(self, index):
        """Change current episode"""
        if 0 <= index < len(self.episodes_list):
            if self.is_playing:
                self.pause_playback()
            self.current_episode_idx = index
            self.load_episode_data()
    
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
        
        next_frame = self.current_frame + self.play_skip
        
        if next_frame >= len(self.image_files):
            self.pause_playback()
            self.statusBar().showMessage("Playback finished")
            return
        
        self.frame_slider.setValue(next_frame)
    
    def update_skip(self, value):
        """Update skip parameter"""
        self.play_skip = value
    
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


def main():
    parser = argparse.ArgumentParser(description="Interact360 Visualizer")
    parser.add_argument("--directory", "-d", type=str, default="./demo/output/",
                       help="Output directory containing processed data (e.g., demo/output)")
    parser.add_argument("--recording_name", "-rn", type=str, default="",
                       help="Recording name to visualize, if not provided will visualize all recordings in the directory")
    
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
