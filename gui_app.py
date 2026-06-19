"""
🧠 Dementia Detection System - Python GUI Application
Native desktop application with real-time eye tracking and blink rate analysis
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import mediapipe as mp
import random

def ear_to_percent(ear):
    """Convert Eye Aspect Ratio to an intuitive 0-100% scale."""
    return max(0.0, min(100.0, ((ear - 0.15) / (0.38 - 0.15)) * 100.0))

def smooth_data(data, window_size=5):
    """Applies a moving average to smooth noisy signal data."""
    if not data or len(data) < window_size:
        return data
    window = np.ones(window_size) / float(window_size)
    smoothed = np.convolve(data, window, 'valid')
    
    pad_start = (window_size - 1) // 2
    pad_end = window_size - 1 - pad_start
    
    # Pad the ends with original data to retain alignment
    if pad_end > 0:
        return data[:pad_start] + smoothed.tolist() + data[-pad_end:]
    return data[:pad_start] + smoothed.tolist()

# Import GUI adapter
from gui_adapter import GUIDementiaAnalyzer

# Import unified risk scoring (gaze score + overall fusion)
from scoring import compute_gaze_score, compute_overall_score, explain_scores, risk_level as _risk_level, load_modality_model, ml_modality_score


class DementiaDetectionGUI:
    """Main GUI Application for Dementia Detection System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Cognitive Risk Assessment")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f4f8')
        
        # Initialize analyzer
        self.analyzer = GUIDementiaAnalyzer()
        self.is_running = False
        self.video_thread = None
        self.update_thread = None
        
        # Video capture
        self.cap = None
        self.current_frame = None
        
        # Session enhancements
        self.session_start_time = None
        self.auto_stop_enabled = False
        self.video_writer = None
        self.video_filename = None
        
        # Patient data container
        self.patient_data = {}
        self.final_metrics = {}
        
        # Setup UI
        self.setup_ui()
        
        # Start update loop
        self.update_metrics_loop()
        
    def setup_ui(self):
        """Setup all UI components"""
        
        # Header
        header = tk.Frame(self.root, bg='#2e8b8b', height=100)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header, 
            text="🧠 Cognitive Risk Assessment",
            font=("Segoe UI", 28, "bold"),
            bg='#2e8b8b',
            fg='white'
        )
        title.pack(pady=(20, 0))
        
        subtitle = tk.Label(
            header,
            text="Real-time Eye Tracking & Blink Rate Analysis",
            font=("Segoe UI", 14),
            bg='white',
            fg='#2e8b8b'
        )
        subtitle.pack()
        
        # Status bar (clean flat look)
        self.status_frame = tk.Frame(self.root, bg='white', height=50, highlightbackground="#e2e8f0", highlightthickness=1)
        self.status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="System Ready - Click Start to Begin",
            font=("Segoe UI", 12),
            bg='white',
            fg='#333'
        )
        self.status_label.pack(pady=10)
        
        # Main content area wrapped in Canvas for scrolling
        main_outer_frame = tk.Frame(self.root, bg='#f0f4f8')
        main_outer_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(main_outer_frame, bg='#f0f4f8', highlightthickness=0)
        self.scrollbar = tk.Scrollbar(main_outer_frame, orient="vertical", command=self.canvas.yview)
        
        # Create scrollable frame inside canvas
        main_frame = tk.Frame(self.canvas, bg='#f0f4f8')
        
        # Bind configure events
        main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind mouse wheel for scrolling
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Setup the 3 Pages
        self.page1_frame = tk.Frame(main_frame, bg='#f0f4f8')
        self.page2_frame = tk.Frame(main_frame, bg='#f0f4f8')
        self.page3_frame = tk.Frame(main_frame, bg='#f0f4f8')  # Gaze Experiment
        self.page4_frame = tk.Frame(main_frame, bg='#f0f4f8')  # Final Results
        
        # --- Page 1: Registration ---
        self.setup_registration_panel(self.page1_frame)
        self.page1_frame.grid_rowconfigure(0, weight=1)
        self.page1_frame.grid_columnconfigure(0, weight=1)
        
        # --- Page 2: Video & Real-time Metrics ---
        self.setup_video_panel(self.page2_frame)
        self.setup_metrics_panel(self.page2_frame)
        self.setup_graphs_panel(self.page2_frame)
        
        # Ensure Page 2 grids expand properly!
        self.page2_frame.grid_columnconfigure(0, weight=2)
        self.page2_frame.grid_columnconfigure(1, weight=1)
        self.page2_frame.grid_rowconfigure(0, weight=1)
        self.page2_frame.grid_rowconfigure(1, weight=1)
        
        # --- Page 3: Results & Graphs ---
        self.setup_gaze_panel(self.page3_frame)
        self.setup_results_panel(self.page4_frame)
        
        # Show initial page
        self.show_page(self.page1_frame)
        
    def show_page(self, frame_to_show):
        for frame in [self.page1_frame, self.page2_frame, self.page3_frame, self.page4_frame]:
            frame.pack_forget()

        self.root.update_idletasks()
    
        frame_to_show.pack(fill=tk.BOTH, expand=True)
    
    # Force grid geometry to re-evaluate inside page2
        if frame_to_show == self.page2_frame:
            self.page2_frame.grid_columnconfigure(0, weight=2)
            self.page2_frame.grid_columnconfigure(1, weight=1)
            self.page2_frame.grid_rowconfigure(0, weight=1)
            self.page2_frame.grid_rowconfigure(1, weight=1)
    
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
        if frame_to_show == self.page1_frame:
            self.status_label.config(text="Patient Registration - Please enter details", bg='white', fg='#333')
            self.status_frame.config(bg='white')
        elif frame_to_show == self.page2_frame:
            self.status_label.config(text="System Ready - Click Start to Begin Detection", bg='white', fg='#333')
            self.status_frame.config(bg='white')
        elif frame_to_show == self.page3_frame:
            self.status_label.config(text="Session Complete - View Results and Export", bg='#11998e', fg='white')
            self.status_frame.config(bg='#11998e')
            
    def setup_registration_panel(self, parent):
        """Setup Patient Registration Form"""
        reg_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        reg_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title = tk.Label(
            reg_frame,
            text="👤 Patient Registration",
            font=("Segoe UI", 20, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        title.pack(pady=20)
        
        form_frame = tk.Frame(reg_frame, bg='white')
        form_frame.pack(pady=10)
        
        # Styling for labels and entries
        lbl_font = ("Segoe UI", 12, "bold")
        entry_font = ("Segoe UI", 12)
        
        self.entries = {}
        fields = [
            ("Token Number:", "token"),
            ("Full Name:", "name"),
            ("Age:", "age"),
            ("Phone Number:", "phone"),
            ("Place/City:", "place"),
            ("Address:", "address")
        ]
        
        for i, (label_text, key) in enumerate(fields):
            tk.Label(form_frame, text=label_text, font=lbl_font, bg='white', fg='#4a5568', width=15, anchor="e").grid(row=i, column=0, pady=10, padx=10)
            entry = tk.Entry(form_frame, font=entry_font, width=30, highlightthickness=1, highlightbackground="#ccd")
            entry.grid(row=i, column=1, pady=10, padx=10)
            self.entries[key] = entry
            
        def submit_form():
            # Validate and save details
            for key, entry in self.entries.items():
                val = entry.get().strip()
                if not val:
                    messagebox.showerror("Validation Error", "All fields are required.")
                    return
                self.patient_data[key] = val
                
            self.show_page(self.page2_frame)
            
        btn = tk.Button(
            reg_frame,
            text="Proceed to Detection",
            font=("Segoe UI", 14, "bold"),
            bg='#2e8b8b',
            fg='white',
            width=25,
            height=2,
            command=submit_form,
            cursor='hand2',
            relief=tk.FLAT,
            activebackground='#206969',
            activeforeground='white'
        )
        btn.pack(pady=30)
        
    def setup_video_panel(self, parent):
        """Setup video feed panel"""
        video_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        video_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        
        # Title
        title = tk.Label(
            video_frame,
            text="📹 Live Video Feed",
            font=("Segoe UI", 16, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        title.pack(pady=10)
        
        # Video display
        self.video_label = tk.Label(video_frame, bg='#1a202c')
        self.video_label.pack(padx=20, pady=10)
        
        # Set default image (smaller, 480x360 to save space)
        default_img = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(default_img, "Click Start Detection to begin", (40, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.update_video_frame(default_img)
        
        # Control buttons
        btn_frame = tk.Frame(video_frame, bg='white')
        btn_frame.pack(pady=15)
        
        def create_button(text, bg_color, hover_color, command, state=tk.NORMAL):
            btn = tk.Button(
                btn_frame,
                text=text,
                font=("Segoe UI", 11, "bold"),
                bg=bg_color,
                fg='white',
                width=18,
                height=2,
                command=command,
                cursor='hand2',
                relief=tk.FLAT,
                bd=0,
                activebackground=hover_color,
                activeforeground='white',
                state=state
            )
            
            # Add hover effects
            def on_enter(e):
                if btn['state'] == tk.NORMAL:
                    btn.config(bg=hover_color)
            def on_leave(e):
                if btn['state'] == tk.NORMAL:
                    btn.config(bg=bg_color)
                    
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            return btn

        self.start_btn = create_button(
            "▶ START DETECTION", 
            '#2e8b8b', '#206969', 
            self.start_detection
        )
        self.start_btn.grid(row=0, column=0, padx=8)
        
        self.stop_btn = create_button(
            "⏹ STOP DETECTION", 
            '#e74c3c', '#c0392b', 
            self.stop_detection, 
            tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=1, padx=8)
        
        self.export_btn = create_button(
            "FINISH & VIEW RESULTS", 
            '#4eb2f7', '#2980b9', 
            self.finish_detection
        )
        self.export_btn.grid(row=0, column=2, padx=8)
        
        self.continue_btn = create_button(
            "⏭ CONTINUE", 
            '#f39c12', '#d68910', 
            self.continue_detection,
            tk.NORMAL
        )
        self.continue_btn.grid(row=0, column=3, padx=8)

        
        self.page2_frame.grid_columnconfigure(0, weight=2)
        self.page2_frame.grid_columnconfigure(1, weight=1)
        
    def setup_metrics_panel(self, parent):
        """Setup real-time metrics panel"""
        metrics_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        metrics_frame.grid(row=0, column=1, sticky='nsew')
        
        # Title
        title = tk.Label(
            metrics_frame,
            text="📊 Real-time Metrics",
            font=("Segoe UI", 14, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        title.pack(pady=5)
        
        # Risk indicator
        self.risk_frame = tk.Frame(metrics_frame, bg='#11998e', relief=tk.FLAT)
        self.risk_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.risk_label = tk.Label(
            self.risk_frame,
            text="Risk Level: LOW",
            font=("Segoe UI", 14, "bold"),
            bg='#11998e',
            fg='white'
        )
        self.risk_label.pack(pady=5)
        
        self.risk_score_label = tk.Label(
            self.risk_frame,
            text="0%",
            font=("Segoe UI", 26, "bold"),
            bg='#11998e',
            fg='white'
        )
        self.risk_score_label.pack(pady=2)
        
        # Metrics grid
        metrics_grid = tk.Frame(metrics_frame, bg='white')
        metrics_grid.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Metric boxes
        self.blink_rate_label = self.create_metric_box(
            metrics_grid, "Blink Rate", "0", "blinks/min", 0, 0
        )
        self.total_blinks_label = self.create_metric_box(
            metrics_grid, "Total Blinks", "0", "count", 0, 1
        )
        self.eye_openness_label = self.create_metric_box(
            metrics_grid, "Eye Openness", "0.00", "EAR", 1, 0
        )
        self.session_time_label = self.create_metric_box(
            metrics_grid, "Session Time", "0", "seconds", 1, 1
        )
        
        # Risk factors
        risk_factors_frame = tk.Frame(metrics_frame, bg='#f8f9fa', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        risk_factors_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        rf_title = tk.Label(
            risk_factors_frame,
            text="⚠️ Risk Factors Detected",
            font=("Segoe UI", 12, "bold"),
            bg='#f8f9fa',
            fg='#2e8b8b'
        )
        rf_title.pack(pady=5)
        
        self.risk_factors_text = tk.Text(
            risk_factors_frame,
            height=3,
            font=("Segoe UI", 9),
            bg='white',
            fg='#333',
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.risk_factors_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.risk_factors_text.insert('1.0', 'No risk factors detected')
        self.risk_factors_text.config(state=tk.DISABLED)
        
        parent.grid_columnconfigure(1, weight=1)
        
    def create_metric_box(self, parent, title, value, unit, row, col):
        """Create a metric display box"""
        box = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        box.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
        
        title_label = tk.Label(
            box,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg='white',
            fg='#4a5568'
        )
        title_label.pack(pady=(15, 5))
        
        value_label = tk.Label(
            box,
            text=value,
            font=("Segoe UI", 24, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        value_label.pack(pady=0)
        
        unit_label = tk.Label(
            box,
            text=unit,
            font=("Segoe UI", 9),
            bg='white',
            fg='#a0aec0'
        )
        unit_label.pack(pady=(0, 15))
        
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(col, weight=1)
        
        return value_label
        
    def setup_graphs_panel(self, parent):
        """Setup analysis graphs panel"""
        graphs_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        graphs_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(20, 0))
        
        # Title
        title = tk.Label(
            graphs_frame,
            text="📈 Analysis Graphs",
            font=("Segoe UI", 16, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        title.pack(pady=10)
        
        # Graphs container
        graphs_container = tk.Frame(graphs_frame, bg='white')
        graphs_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figures (styled like the web UI)
        self.fig1 = Figure(figsize=(6, 3), dpi=100)
        self.fig1.patch.set_facecolor('#f4f7fe')
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title('Eye Openness Over Time', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax1.set_ylim(0, 0.5)
        self.ax1.grid(True, alpha=0.3, color='#ccd')
        self.ax1.set_facecolor('#f4f7fe')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_color('#ccd')
        self.ax1.spines['bottom'].set_color('#ccd')
        self.ax1.tick_params(axis='x', colors='#666', labelsize=8)
        self.ax1.tick_params(axis='y', colors='#666', labelsize=8)
        self.fig1.subplots_adjust(bottom=0.25, left=0.15, top=0.8)
        
        self.canvas1 = FigureCanvasTkAgg(self.fig1, graphs_container)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.fig2 = Figure(figsize=(6, 3), dpi=100)
        self.fig2.patch.set_facecolor('#f4f7fe')
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title('Blink Pattern', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax2.set_ylim(0, 1.0)
        self.ax2.grid(True, alpha=0.3, color='#ccd')
        self.ax2.set_facecolor('#f4f7fe')
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_color('#ccd')
        self.ax2.spines['bottom'].set_color('#ccd')
        self.ax2.tick_params(axis='x', colors='#666', labelsize=8)
        self.ax2.tick_params(axis='y', colors='#666', labelsize=8)
        self.fig2.subplots_adjust(bottom=0.25, left=0.15, top=0.8)
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, graphs_container)
        self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        parent.grid_rowconfigure(1, weight=1)

    def setup_gaze_panel(self, parent):
        """Setup Page 3: Gaze Stimulus Experiment"""
        gaze_main_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        gaze_main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(
            gaze_main_frame,
            text="👁️ Gaze Stimulus Experiment",
            font=("Segoe UI", 20, "bold"),
            bg='white',
            fg='#2e8b8b'
        )
        title.pack(pady=10)
        
        content_frame = tk.Frame(gaze_main_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Side: Video Output & Controls
        left_frame = tk.Frame(content_frame, bg='white')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.gaze_video_label = tk.Label(left_frame, bg='black')
        self.gaze_video_label.pack(pady=10)
        
        # Default placeholder image
        default_img = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(default_img, "Ready for Gaze Experiment", (40, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Convert to PIL Image
        img = Image.fromarray(default_img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.gaze_video_label.imgtk = imgtk
        self.gaze_video_label.configure(image=imgtk)

        btn_frame = tk.Frame(left_frame, bg='white')
        btn_frame.pack(pady=10)

        def create_button(parent, text, bg_color, hover_color, command, state=tk.NORMAL):
            btn = tk.Button(
                parent,
                text=text,
                font=("Segoe UI", 11, "bold"),
                bg=bg_color,
                fg='white',
                width=18,
                height=2,
                command=command,
                cursor='hand2',
                relief=tk.FLAT,
                bd=0,
                activebackground=hover_color,
                activeforeground='white',
                state=state
            )
            def on_enter(e):
                if btn['state'] == tk.NORMAL:
                    btn.config(bg=hover_color)
            def on_leave(e):
                if btn['state'] == tk.NORMAL:
                    btn.config(bg=bg_color)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            return btn
            
        self.start_gaze_btn = create_button(btn_frame, "▶ START EXPERIMENT", '#2e8b8b', '#206969', self.start_gaze_experiment)
        self.start_gaze_btn.grid(row=0, column=0, padx=5)
        
        self.stop_gaze_btn = create_button(btn_frame, "⏹ STOP EXPERIMENT", '#e74c3c', '#c0392b', self.stop_gaze_experiment, tk.DISABLED)
        self.stop_gaze_btn.grid(row=0, column=1, padx=5)

        self.finish_gaze_btn = create_button(btn_frame, "FINISH EXPERIMENT", '#4eb2f7', '#2980b9', self.finish_gaze_experiment, tk.DISABLED)
        self.finish_gaze_btn.grid(row=0, column=2, padx=5)

        self.gaze_voice_btn = create_button(btn_frame, "🎙 VOICE ANALYSIS", '#7b2dbe', '#5a1fa0', self.start_voice_analysis)
        self.gaze_voice_btn.grid(row=0, column=3, padx=5)

        # Right Side: Metrics
        right_frame = tk.Frame(content_frame, bg='white')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        metrics_grid = tk.Frame(right_frame, bg='white')
        metrics_grid.pack(fill=tk.X, pady=10)

        self.gaze_dir_label = self.create_metric_box(metrics_grid, "Current Gaze", "CENTER", "Direction", 0, 0)
        self.reaction_time_label = self.create_metric_box(metrics_grid, "Reaction Time", "0.00", "Seconds", 0, 1)
        self.saccade_speed_label = self.create_metric_box(metrics_grid, "Saccade Speed", "0.00", "px/sec", 1, 0)
        self.trial_num_label = self.create_metric_box(metrics_grid, "Trial Number", "0 / 10", "Count", 1, 1)

        # Live per-trial gaze graphs (reaction time + saccade speed)
        gaze_graph_frame = tk.Frame(right_frame, bg='white')
        gaze_graph_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 10))
        self.fig_gaze = Figure(figsize=(5.2, 4.2), dpi=80)
        self.ax_gaze_rt = self.fig_gaze.add_subplot(211)
        self.ax_gaze_ss = self.fig_gaze.add_subplot(212)
        self.fig_gaze.subplots_adjust(hspace=0.55, left=0.15, right=0.96, top=0.90, bottom=0.13)
        self.canvas_gaze = FigureCanvasTkAgg(self.fig_gaze, gaze_graph_frame)
        self.canvas_gaze.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._last_gaze_plot_n = -1
        self._init_gaze_graphs()

        self.gaze_cap = None
        self.gaze_running = False
        self.gaze_results = {
            "avg_reaction_time": 0.0,
            "avg_saccade_speed": 0.0,
            "accuracy": 0,
            "trials": 0,
            "reaction_times": [],
            "saccade_speeds": []
        }
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
    def setup_results_panel(self, parent):
        """Setup Page 3: Final Results and Export Panel"""
        # Top panel for patient details
        details_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        details_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title = tk.Label(details_frame, text="📋 Session Summary & Patient Details", font=("Segoe UI", 16, "bold"), bg='white', fg='#2e8b8b')
        title.pack(pady=10)
        
        self.patient_details_text = tk.Text(details_frame, height=4, font=("Segoe UI", 11), bg='#f8f9fa', fg='#333', relief=tk.FLAT)
        self.patient_details_text.pack(fill=tk.X, padx=20, pady=10)
        self.patient_details_text.config(state=tk.DISABLED)
        
        # Middle panel for final metrics
        metrics_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.final_metrics_text = tk.Text(metrics_frame, height=7, font=("Segoe UI", 12, "bold"), bg='#f8f9fa', fg='#2e8b8b', relief=tk.FLAT)
        self.final_metrics_text.pack(fill=tk.X, padx=20, pady=10)
        self.final_metrics_text.config(state=tk.DISABLED)
        
        # Explainable-AI panel: which feature contributed how much to the final score
        explain_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, highlightbackground="#e2e8f0", highlightthickness=1)
        explain_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(explain_frame, text="🔍 Why this score? (Explainable AI)",
                 font=("Segoe UI", 16, "bold"), bg='white', fg='#2e8b8b').pack(pady=(10, 4))
        
        explain_body = tk.Frame(explain_frame, bg='white')
        explain_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Left: contribution bar chart
        chart_holder = tk.Frame(explain_body, bg='white')
        chart_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.fig_explain = Figure(figsize=(5.2, 3.2), dpi=80)
        self.ax_explain = self.fig_explain.add_subplot(111)
        self.canvas_explain = FigureCanvasTkAgg(self.fig_explain, chart_holder)
        self.canvas_explain.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: plain-language explanation text
        text_holder = tk.Frame(explain_body, bg='white')
        text_holder.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.explain_text = tk.Text(text_holder, width=46, font=("Segoe UI", 9),
                                    bg='#f8f9fa', fg='#333', relief=tk.FLAT, wrap=tk.WORD)
        self.explain_text.pack(fill=tk.BOTH, expand=True)
        self.explain_text.config(state=tk.DISABLED)
        
        # Bottom panel for Graphs view (recycled from page 2 old setup)
        graphs_frame = tk.Frame(parent, bg='white')
        graphs_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add inner frames for grid
        frame_row1 = tk.Frame(graphs_frame, bg='white')
        frame_row1.pack(fill=tk.BOTH, expand=True)
        frame_row2 = tk.Frame(graphs_frame, bg='white')
        frame_row2.pack(fill=tk.BOTH, expand=True)
        
        frame_graph1 = tk.Frame(frame_row1, bg='white')
        frame_graph1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        frame_graph2 = tk.Frame(frame_row1, bg='white')
        frame_graph2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        frame_graph3 = tk.Frame(frame_row2, bg='white')
        frame_graph3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        frame_graph4 = tk.Frame(frame_row2, bg='white')
        frame_graph4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Setup 4 figures
        self.fig_final1 = Figure(figsize=(5, 3), dpi=80)
        self.ax_final1 = self.fig_final1.add_subplot(111)
        self.canvas_final1 = FigureCanvasTkAgg(self.fig_final1, frame_graph1)
        self.canvas_final1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig_final2 = Figure(figsize=(5, 3), dpi=80)
        self.ax_final2 = self.fig_final2.add_subplot(111)
        self.canvas_final2 = FigureCanvasTkAgg(self.fig_final2, frame_graph2)
        self.canvas_final2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig_final3 = Figure(figsize=(5, 3), dpi=80)
        self.ax_final3 = self.fig_final3.add_subplot(111)
        self.canvas_final3 = FigureCanvasTkAgg(self.fig_final3, frame_graph3)
        self.canvas_final3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig_final4 = Figure(figsize=(5, 3), dpi=80)
        self.ax_final4 = self.fig_final4.add_subplot(111)
        self.canvas_final4 = FigureCanvasTkAgg(self.fig_final4, frame_graph4)
        self.canvas_final4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control Buttons
        btn_frame = tk.Frame(parent, bg='#f0f4f8')
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="📥 EXPORT TO EXCEL (CSV)", font=("Segoe UI", 12, "bold"), bg='#27ae60', fg='white', width=25, height=2, cursor='hand2', relief=tk.FLAT, command=self.export_patient_data).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="📄 GENERATE PDF REPORT", font=("Segoe UI", 12, "bold"), bg='#8e44ad', fg='white', width=23, height=2, cursor='hand2', relief=tk.FLAT, command=self.generate_report).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="⬅ BACK TO DETECTION", font=("Segoe UI", 12, "bold"), bg='#f39c12', fg='white', width=22, height=2, cursor='hand2', relief=tk.FLAT, command=self.back_to_detection).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="➕ NEW PATIENT", font=("Segoe UI", 12, "bold"), bg='#34495e', fg='white', width=20, height=2, cursor='hand2', relief=tk.FLAT, command=self.reset_new_patient).pack(side=tk.LEFT, padx=10)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def update_video_frame(self, frame):
        """Update video display with new frame"""
        # Convert BGR to RGB
        if len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Resize to fit display (smaller to save vertical space)
        frame_resized = cv2.resize(frame_rgb, (480, 360))
        
        # Convert to PIL Image
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def video_capture_loop(self):
        """Continuous video capture and processing (runs in background thread)"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame with analyzer
                    processed_frame = self.analyzer.process_frame(frame)
                    
                    # Store frame for main thread to pick up
                    self.current_frame = processed_frame
                    
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
    def update_video_ui_loop(self):
        """Update video display on the main Tkinter thread"""
        if self.is_running and self.current_frame is not None:
            self.update_video_frame(self.current_frame)
            
            # Record frame to video file
            if self.video_writer is not None:
                self.video_writer.write(self.current_frame)
            
        # Schedule next update (approx 30 FPS)
        if self.is_running:
            self.root.after(33, self.update_video_ui_loop)
            
    def update_metrics_loop(self):
        """Update metrics display every second"""
        if self.is_running:
            metrics = self.analyzer.get_metrics()
            
            # Update metric values
            self.blink_rate_label.config(text=f"{metrics['blink_rate']:.1f}")
            self.total_blinks_label.config(text=str(metrics['total_blinks']))
            self.eye_openness_label.config(text=f"{ear_to_percent(metrics['avg_ear']):.0f}%")
            self.session_time_label.config(text=str(int(metrics['session_duration'])))
            
            # Update risk indicator
            risk_level = metrics['risk_level']
            risk_score = metrics['risk_score']
            
            self.risk_label.config(text=f"Risk Level: {risk_level}")
            self.risk_score_label.config(text=f"{risk_score:.0f}%")
            
            # Update risk indicator color
            risk_colors = {
                'LOW': '#11998e',
                'MILD': '#f2994a',
                'MODERATE': '#ff9a56',
                'HIGH': '#eb3349',
                'VERY HIGH': '#c94b4b'
            }
            color = risk_colors.get(risk_level, '#11998e')
            self.risk_frame.config(bg=color)
            self.risk_label.config(bg=color)
            self.risk_score_label.config(bg=color)
            
            # Update risk factors
            risk_factors = metrics.get('risk_factors', [])
            self.risk_factors_text.config(state=tk.NORMAL)
            self.risk_factors_text.delete('1.0', tk.END)
            if risk_factors:
                for factor in risk_factors:
                    self.risk_factors_text.insert(tk.END, f"• {factor}\n")
            else:
                self.risk_factors_text.insert('1.0', 'No risk factors detected')
            self.risk_factors_text.config(state=tk.DISABLED)
            
            # Update graphs
            self.update_graphs(metrics)
            
            # Check for 60-second auto-stop
            if self.auto_stop_enabled and self.session_start_time is not None:
                elapsed_time = time.time() - self.session_start_time
                if elapsed_time >= 60.0:
                    self.auto_stop_enabled = False
                    self.stop_detection()
                    # Store the blink result now, but DO NOT jump to results, so
                    # the gaze and voice tests can still be run.
                    try:
                        self.final_metrics = self.analyzer.get_metrics()
                    except Exception:
                        pass
                    messagebox.showinfo(
                        "Blink Analysis Complete",
                        "60-second blink analysis complete and saved.\n\n"
                        "Click CONTINUE to run the Gaze test, use VOICE ANALYSIS,\n"
                        "or click FINISH & VIEW RESULTS to see the results now.")
            
        # Schedule next update
        self.root.after(1000, self.update_metrics_loop)
        
    def update_graphs(self, metrics):
        """Update analysis graphs"""
        # Update EAR graph
        ear_history = metrics.get('ear_history', [])
        partial_blinks = metrics.get('partial_blinks', [])
        if ear_history:
            ear_data = ear_history[-50:]
            # Smooth the real-time data slightly (window 3-5 frames)
            left_raw = [ear_to_percent(e['left']) for e in ear_data]
            right_raw = [ear_to_percent(e['right']) for e in ear_data]
            
            left_values = smooth_data(left_raw, window_size=5)
            right_values = smooth_data(right_raw, window_size=5)
            
            ear_times = [datetime.fromtimestamp(e['timestamp']).strftime('%H:%M:%S') for e in ear_data]
            frames = list(range(len(ear_data)))
            
            self.ax1.clear()
            # Plot Left and Right EAR
            self.ax1.plot(frames, left_values, color='#3498db', linewidth=1.5, label='Left Eye EAR')
            self.ax1.plot(frames, right_values, color='#9b59b6', linewidth=1.5, label='Right Eye EAR')
            
            # Plot threshold
            thresh_percent = ear_to_percent(metrics.get('current_threshold', 0.25))
            self.ax1.axhline(y=thresh_percent, color='red', linestyle='--', linewidth=1.5, label='Adaptive Blink Threshold')
            
            # Draw Micro-Sleep Warning Zones (Red Axvspans)
            micro_sleeps = metrics.get('micro_sleeps', [])
            if micro_sleeps:
                min_time = ear_data[0]['timestamp']
                max_time = ear_data[-1]['timestamp']
                for ms in micro_sleeps:
                    # Only draw if the micro-sleep overlaps with the current 50-frame window
                    if ms['end'] >= min_time and ms['start'] <= max_time:
                        start_idx = min(range(len(ear_data)), key=lambda i: abs(ear_data[i]['timestamp'] - max(ms['start'], min_time)))
                        end_idx = min(range(len(ear_data)), key=lambda i: abs(ear_data[i]['timestamp'] - min(ms['end'], max_time)))
                        if start_idx != end_idx:
                            self.ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3, label='Micro-Sleep (>0.5s)' if 'Micro-Sleep (>0.5s)' not in self.ax1.get_legend_handles_labels()[1] else "")
            
            # Mark Partial Blinks if they fall inside the current x-axis window
            if partial_blinks:
                min_time = ear_data[0]['timestamp']
                max_time = ear_data[-1]['timestamp']
                
                pb_x = []
                pb_y = []
                for pb in partial_blinks:
                    if min_time <= pb['timestamp'] <= max_time:
                        # Find closest frame index
                        closest_idx = min(range(len(ear_data)), key=lambda i: abs(ear_data[i]['timestamp'] - pb['timestamp']))
                        pb_x.append(closest_idx)
                        pb_y.append(ear_to_percent(pb['min_ear']))
                
                if pb_x:
                    self.ax1.scatter(pb_x, pb_y, color='#e67e22', marker='^', s=40, zorder=5, label='Partial Blink')
            
            self.ax1.set_title('Eye Openness Over Time (L/R Split)', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
            self.ax1.set_ylim(0, 100)
            self.ax1.set_ylabel('Eye openness (%)', fontsize=9, color='#555')
            self.ax1.grid(True, alpha=0.3, color='#ccd')
            
            # Re-apply styling that gets cleared
            self.ax1.spines['top'].set_visible(False)
            self.ax1.spines['right'].set_visible(False)
            self.ax1.spines['left'].set_color('#ccd')
            self.ax1.spines['bottom'].set_color('#ccd')
            
            # Set rotated x ticks
            self.ax1.set_xticks(frames[::2])
            self.ax1.set_xticklabels(ear_times[::2], rotation=45, ha='right')
            
            self.ax1.legend(loc='upper right', frameon=False, fontsize=8)
            self.canvas1.draw()
            
        # Update blink graph
        blink_times = metrics.get('blink_times', [])
        if len(blink_times) > 1:
            recent_blinks = blink_times[-21:]  # Get up to 21 to have 20 intervals
            intervals = np.diff(recent_blinks)
            self.ax2.clear()
            
            x_pos = np.arange(len(intervals))
            colors = ['#2ecc71' if 2.0 <= interval <= 6.0 else '#e74c3c' for interval in intervals]
            
            # Shade the normal inter-blink interval band (2-6 s) for context
            self.ax2.axhspan(2.0, 6.0, color='#2ecc71', alpha=0.10, label='Normal (2-6s)')
            self.ax2.bar(x_pos, intervals, color=colors, alpha=0.9, width=0.7, label='Interval (s)')
            
            self.ax2.set_title('Blink Regularity (Interval)', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
            self.ax2.set_ylabel('Interval (s)', fontsize=9, color='#555')
            self.ax2.set_ylim(0, max(5.0, max(intervals) * 1.2))
            self.ax2.grid(True, alpha=0.3, color='#ccd')
            
            # Re-apply styling
            self.ax2.spines['top'].set_visible(False)
            self.ax2.spines['right'].set_visible(False)
            self.ax2.spines['left'].set_color('#ccd')
            self.ax2.spines['bottom'].set_color('#ccd')
            
            # Setup x-axis labels
            labels = [f"Blink {i+1}" for i in range(len(intervals))]
            self.ax2.set_xticks(x_pos)
            self.ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
            
            self.ax2.legend(loc='upper right', frameon=False, fontsize=8)
            self.canvas2.draw()
            
    def start_detection(self):
        """Start webcam detection"""
        if not self.is_running:
            # Try to open webcam with CAP_DSHOW (best for Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(0)
                
            # If still not open, try index 1
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)

            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam! Please check if another app is using it.")
                return
                
            # Setup video writer
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Default fallback for missing values
            if width == 0 or height == 0:
                 width, height = 640, 480
            if fps == 0 or fps != fps:
                 fps = 30.0
                 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"session_video_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, (width, height))
            
            # Set session timers
            self.session_start_time = time.time()
            self.auto_stop_enabled = True
            
            # Start analyzer
            self.analyzer.start_session()
            self.is_running = True
            
            # Update UI
            self.status_label.config(text="Detection Active - Analyzing...", bg='#11998e', fg='white')
            self.status_frame.config(bg='#11998e')
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.continue_btn.config(state=tk.NORMAL)
            
            # Start video thread for processing
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.video_thread.start()
            
            # Start UI update loop for video
            self.update_video_ui_loop()
            
    def stop_detection(self):
        """Stop webcam detection"""
        if self.is_running:
            self.is_running = False
            
            # Stop analyzer
            self.analyzer.stop_session()
            
            # Release webcam
            if self.cap:
                self.cap.release()
                
            # Update UI
            self.status_label.config(text="Detection Stopped", bg='white', fg='#333')
            self.status_frame.config(bg='white')
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.continue_btn.config(state=tk.NORMAL)
            
            # Release video writer
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
    def continue_detection(self):
        """Stop blink detection and go to Gaze Stimulus Experiment (Page 3)"""
        if self.is_running:
            self.stop_detection()
        
        # Move to Page 3
        self.show_page(self.page3_frame)
        
    def get_ratio(self, mesh_points):
        """Calculate eye ratio from face mesh landmarks"""
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        
        left_iris = mesh_points[LEFT_IRIS]
        right_iris = mesh_points[RIGHT_IRIS]
        
        left_center = np.mean(left_iris, axis=0)
        right_center = np.mean(right_iris, axis=0)
        
        left_ratio = (left_center[0] - mesh_points[33][0]) / (mesh_points[133][0] - mesh_points[33][0] + 1e-6)
        right_ratio = (right_center[0] - mesh_points[362][0]) / (mesh_points[263][0] - mesh_points[362][0] + 1e-6)
        
        return (left_ratio + right_ratio) / 2
        
    def start_gaze_experiment(self):
        """Start gaze experiment"""
        if self.gaze_running: return
        
        self.gaze_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.gaze_cap.isOpened():
            self.gaze_cap = cv2.VideoCapture(0)
            if not self.gaze_cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam for gaze tracking!")
                return
                
        self.gaze_running = True
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        self.gaze_results = {
            "avg_reaction_time": 0.0,
            "avg_saccade_speed": 0.0,
            "accuracy": 0,
            "trials": 0,
            "reaction_times": [],
            "saccade_speeds": []
        }
        self._last_gaze_plot_n = -1
        
        self.start_gaze_btn.config(state=tk.DISABLED)
        self.stop_gaze_btn.config(state=tk.NORMAL)
        self.finish_gaze_btn.config(state=tk.DISABLED)
        
        self.gaze_calibration_mode = True
        self.gaze_calibration_stage = 0 # 0=LEFT, 1=CENTER, 2=RIGHT
        self.gaze_calibration_data = {"LEFT": [], "CENTER": [], "RIGHT": []}
        self.gaze_stage_time = time.time()
        
        self.gaze_experiment_state = "CALIBRATION"
        
        self.trial_count = 0
        self.total_trials = 10
        self.correct_trials = 0
        self.current_stimulus = None
        self.stimulus_time = 0
        self.fixation_time = 0
        self.prev_eye_x = None
        self.prev_time = None
        
        self.gaze_thread = threading.Thread(target=self.gaze_capture_loop, daemon=True)
        self.gaze_thread.start()
        
        self.update_gaze_ui_loop()

    def stop_gaze_experiment(self):
        """Stop gaze experiment prematurely"""
        self.gaze_running = False
        if self.gaze_cap:
            self.gaze_cap.release()
            
        self.start_gaze_btn.config(state=tk.NORMAL)
        self.stop_gaze_btn.config(state=tk.DISABLED)
        self.finish_gaze_btn.config(state=tk.NORMAL)

    def gaze_capture_loop(self):
        """Capture and process gaze frames"""
        while self.gaze_running:
            if self.gaze_cap and self.gaze_cap.isOpened():
                ret, frame = self.gaze_cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb)
                    
                    mesh_points = None
                    if results.multi_face_landmarks:
                        mesh_points = np.array(
                            [(int(p.x*w), int(p.y*h)) for p in results.multi_face_landmarks[0].landmark]
                        )
                    
                    if self.gaze_experiment_state == "CALIBRATION":
                        stage_names = ["LEFT", "CENTER", "RIGHT"]
                        if self.gaze_calibration_stage < 3:
                            direction = stage_names[self.gaze_calibration_stage]
                            cv2.putText(frame, f"LOOK {direction}", (w//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            
                            if mesh_points is not None:
                                ratio = self.get_ratio(mesh_points)
                                self.gaze_calibration_data[direction].append(ratio)
                                
                            if time.time() - self.gaze_stage_time > 3:
                                self.gaze_calibration_stage += 1
                                self.gaze_stage_time = time.time()
                        else:
                            # Calibration Complete
                            self.left_thresh = np.mean(self.gaze_calibration_data["LEFT"])
                            self.center_thresh = np.mean(self.gaze_calibration_data["CENTER"])
                            self.right_thresh = np.mean(self.gaze_calibration_data["RIGHT"])
                            
                            self.gaze_experiment_state = "FIXATION"
                            self.fixation_time = time.time()
                            
                    elif self.gaze_experiment_state == "FIXATION":
                        cv2.putText(frame, "+", (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                        if time.time() - self.fixation_time > random.uniform(1.5, 2.5):
                            self.current_stimulus = random.choice(["LEFT", "RIGHT"])
                            self.stimulus_time = time.time()
                            self.gaze_experiment_state = "STIMULUS"
                            
                    elif self.gaze_experiment_state == "STIMULUS":
                        if self.current_stimulus == "LEFT":
                            cv2.circle(frame, (80, h//2), 40, (0, 0, 255), -1)
                        if self.current_stimulus == "RIGHT":
                            cv2.circle(frame, (w - 80, h//2), 40, (0, 0, 255), -1)
                            
                        if mesh_points is not None:
                            ratio = self.get_ratio(mesh_points)
                            gaze_dir = "CENTER"
                            if ratio < (self.left_thresh + self.center_thresh) / 2:
                                gaze_dir = "LEFT"
                            elif ratio > (self.right_thresh + self.center_thresh) / 2:
                                gaze_dir = "RIGHT"
                                
                            eye_x = np.mean([mesh_points[474][0], mesh_points[469][0]])
                            current_time = time.time()
                            speed = 0
                            
                            if self.prev_eye_x is not None:
                                dist = abs(eye_x - self.prev_eye_x)
                                dt = current_time - self.prev_time
                                if dt > 0:
                                    speed = dist / dt
                                    self.gaze_results["saccade_speeds"].append(speed)
                                    self.last_saccade_speed = speed
                                    
                            self.prev_eye_x = eye_x
                            self.prev_time = current_time
                            
                            self.current_gaze_dir = gaze_dir
                            
                            if gaze_dir == self.current_stimulus:
                                reaction = time.time() - self.stimulus_time
                                self.gaze_results["reaction_times"].append(reaction)
                                self.last_reaction_time = reaction
                                
                                self.trial_count += 1
                                self.correct_trials += 1
                                
                                if self.trial_count >= self.total_trials:
                                    self.gaze_running = False
                                    self.gaze_experiment_state = "FINISHED"
                                else:
                                    self.gaze_experiment_state = "FIXATION"
                                    self.fixation_time = time.time()
                    
                    self.current_gaze_frame = frame
                    
            time.sleep(0.01)

    def update_gaze_ui_loop(self):
        """Update Gaze UI with frames and metrics"""
        if hasattr(self, 'current_gaze_frame') and self.current_gaze_frame is not None:
            # Update video label
            frame_rgb = cv2.cvtColor(self.current_gaze_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (480, 360))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.gaze_video_label.imgtk = imgtk
            self.gaze_video_label.configure(image=imgtk)
            
        if hasattr(self, 'current_gaze_dir'):
            self.gaze_dir_label.config(text=self.current_gaze_dir)
            
        if hasattr(self, 'last_reaction_time'):
            self.reaction_time_label.config(text=f"{self.last_reaction_time:.2f}")
            
        if hasattr(self, 'last_saccade_speed'):
            self.saccade_speed_label.config(text=f"{self.last_saccade_speed:.0f}")
            
        if hasattr(self, 'trial_count'):
            self.trial_num_label.config(text=f"{self.trial_count} / {self.total_trials}")

        # Refresh the live gaze graphs only when a new trial has completed
        n_done = len(self.gaze_results.get("reaction_times", []))
        if n_done != self._last_gaze_plot_n:
            self._last_gaze_plot_n = n_done
            try:
                self.update_gaze_graphs()
            except Exception:
                pass
            
        if not self.gaze_running and getattr(self, 'gaze_experiment_state', '') == "FINISHED":
            self.finish_gaze_btn.config(state=tk.NORMAL)
            self.stop_gaze_btn.config(state=tk.DISABLED)
            
        if self.gaze_running or getattr(self, 'gaze_experiment_state', '') == "FINISHED":
            self.root.after(33, self.update_gaze_ui_loop)

    def _init_gaze_graphs(self):
        """Draw the empty gaze graphs with healthy-range guides before any trials."""
        for ax in (self.ax_gaze_rt, self.ax_gaze_ss):
            ax.clear()
        self.ax_gaze_rt.set_title("Reaction Time per Trial", fontsize=10, fontweight='bold', color='#555')
        self.ax_gaze_rt.axhspan(0, 0.5, color='#2ecc71', alpha=0.12)
        self.ax_gaze_rt.axhline(0.5, color='#2ecc71', ls='--', lw=1)
        self.ax_gaze_rt.text(0.02, 0.5, "healthy \u2264 0.5 s", fontsize=7, color='#2ecc71',
                             transform=self.ax_gaze_rt.transAxes, va='bottom')
        self.ax_gaze_rt.set_ylabel("seconds", fontsize=8)
        self.ax_gaze_rt.set_xlabel("trial", fontsize=8)

        self.ax_gaze_ss.set_title("Saccade Speed (movement samples)", fontsize=10, fontweight='bold', color='#555')
        self.ax_gaze_ss.axhline(300, color='#2ecc71', ls='--', lw=1)
        self.ax_gaze_ss.text(0.02, 0.86, "healthy \u2265 300 px/s", fontsize=7, color='#2ecc71',
                             transform=self.ax_gaze_ss.transAxes, va='top')
        self.ax_gaze_ss.set_ylabel("px / s", fontsize=8)
        self.ax_gaze_ss.set_xlabel("sample", fontsize=8)
        for ax in (self.ax_gaze_rt, self.ax_gaze_ss):
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        self.canvas_gaze.draw_idle()

    def update_gaze_graphs(self):
        """Refresh the live gaze graphs from the current trial data."""
        rt = list(self.gaze_results.get("reaction_times", []))
        ss = list(self.gaze_results.get("saccade_speeds", []))
        GAZE = '#f39c12'

        self.ax_gaze_rt.clear()
        self.ax_gaze_rt.set_title("Reaction Time per Trial", fontsize=10, fontweight='bold', color='#555')
        self.ax_gaze_rt.axhspan(0, 0.5, color='#2ecc71', alpha=0.12)
        self.ax_gaze_rt.axhline(0.5, color='#2ecc71', ls='--', lw=1)
        if rt:
            x = list(range(1, len(rt) + 1))
            self.ax_gaze_rt.plot(x, rt, '-o', color=GAZE, lw=1.6, ms=5)
            self.ax_gaze_rt.set_xticks(x)
            avg = sum(rt) / len(rt)
            self.ax_gaze_rt.axhline(avg, color='#c0392b', ls=':', lw=1)
            self.ax_gaze_rt.text(0.98, 0.92, f"avg {avg:.2f}s", fontsize=7, color='#c0392b',
                                 ha='right', va='top', transform=self.ax_gaze_rt.transAxes)
        self.ax_gaze_rt.set_ylabel("seconds", fontsize=8)
        self.ax_gaze_rt.set_xlabel("trial", fontsize=8)

        self.ax_gaze_ss.clear()
        self.ax_gaze_ss.set_title("Saccade Speed (movement samples)", fontsize=10, fontweight='bold', color='#555')
        self.ax_gaze_ss.axhline(300, color='#2ecc71', ls='--', lw=1)
        if ss:
            self.ax_gaze_ss.plot(range(1, len(ss) + 1), ss, color=GAZE, lw=1.0, alpha=0.85)
            avg_ss = sum(ss) / len(ss)
            self.ax_gaze_ss.axhline(avg_ss, color='#c0392b', ls=':', lw=1)
            self.ax_gaze_ss.text(0.98, 0.92, f"avg {avg_ss:.0f} px/s", fontsize=7, color='#c0392b',
                                 ha='right', va='top', transform=self.ax_gaze_ss.transAxes)
        self.ax_gaze_ss.set_ylabel("px / s", fontsize=8)
        self.ax_gaze_ss.set_xlabel("sample", fontsize=8)

        for ax in (self.ax_gaze_rt, self.ax_gaze_ss):
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        self.canvas_gaze.draw_idle()

    def finish_gaze_experiment(self):
        """Calculate averages and transition to results"""
        if self.gaze_running:
            self.stop_gaze_experiment()
            
        # Compile gaze results
        rt = self.gaze_results["reaction_times"]
        ss = self.gaze_results["saccade_speeds"]
        
        self.gaze_results["avg_reaction_time"] = np.mean(rt) if rt else 0.0
        self.gaze_results["avg_saccade_speed"] = np.mean(ss) if ss else 0.0
        self.gaze_results["accuracy"] = (self.correct_trials / max(1, self.trial_count)) * 100
        self.gaze_results["trials"] = self.trial_count
        
        # Compute the gaze risk score (0-100) from the raw metrics
        self.gaze_score = compute_gaze_score(self.gaze_results)
        self.gaze_results["score"] = self.gaze_score["score"]
        self.gaze_results["level"] = self.gaze_score["level"]
        
        # Get final metrics to save for displaying in Page 4
        self.final_metrics = self.analyzer.get_metrics()
        
        # Move UI
        self.show_page(self.page4_frame)
        self.populate_results()
        
    def finish_detection(self):
        """Transition to Page 4 and finalize metrics"""
        if self.is_running:
            self.stop_detection()
            
        # Get final metrics to save for displaying in Page 4
        self.final_metrics = self.analyzer.get_metrics()
        
        # Move UI
        self.show_page(self.page4_frame)
        self.populate_results()
        
    def populate_results(self):
        """Populate Page 4 with final details and graphs"""
        # Update details text
        details = (f"Token: {self.patient_data.get('token', 'N/A')} | "
                   f"Name: {self.patient_data.get('name', 'N/A')} | "
                   f"Age: {self.patient_data.get('age', 'N/A')}\n"
                   f"Phone: {self.patient_data.get('phone', 'N/A')} | "
                   f"Place: {self.patient_data.get('place', 'N/A')}\n"
                   f"Address: {self.patient_data.get('address', 'N/A')}")
        self.patient_details_text.config(state=tk.NORMAL)
        self.patient_details_text.delete('1.0', tk.END)
        self.patient_details_text.insert(tk.END, details)
        self.patient_details_text.config(state=tk.DISABLED)
        
        # Update metrics text
        gaze = getattr(self, 'gaze_results', {})
        avg_rt = gaze.get('avg_reaction_time', 0.0)
        avg_ss = gaze.get('avg_saccade_speed', 0.0)
        acc = gaze.get('accuracy', 0.0)
        trials = gaze.get('trials', 0)

        # --- Per-test scores -------------------------------------------------
        eye_score = self.final_metrics.get('risk_score', 0.0)
        eye_level = self.final_metrics.get('risk_level', 'UNKNOWN')

        # Eye is only counted if the blink test actually produced data
        eye_available = (self.final_metrics.get('total_blinks', 0) > 0
                         or len(self.final_metrics.get('ear_history', []) or []) > 0)

        gaze_score_info = compute_gaze_score(gaze) if gaze else {'available': False}
        gaze_available = gaze_score_info.get('available', False)
        gaze_score = gaze_score_info.get('score', 0.0)
        gaze_level = gaze_score_info.get('level', 'N/A')

        # --- Raw feature vectors (for logging + optional ML models) ----------
        import numpy as _np, os as _os
        _bt = self.final_metrics.get('blink_times', []) or []
        blink_variance = float(_np.var(_np.diff(_bt))) if len(_bt) >= 2 else 0.0
        blink_features = {
            'blink_rate': float(self.final_metrics.get('blink_rate', 0.0)),
            'blink_variance': blink_variance,
            'eye_openness': float(self.final_metrics.get('avg_ear', 0.0)),
            'micro_sleeps': float(len(self.final_metrics.get('micro_sleeps', []) or [])
                                  if isinstance(self.final_metrics.get('micro_sleeps'), list)
                                  else self.final_metrics.get('micro_sleeps', 0) or 0),
            'partial_blinks': float(len(self.final_metrics.get('partial_blinks', []) or [])
                                    if isinstance(self.final_metrics.get('partial_blinks'), list)
                                    else self.final_metrics.get('partial_blinks', 0) or 0),
        }
        gaze_features = {
            'avg_reaction_time': float(gaze.get('avg_reaction_time', 0.0)) if gaze else 0.0,
            'avg_saccade_speed': float(gaze.get('avg_saccade_speed', 0.0)) if gaze else 0.0,
            'accuracy': float(gaze.get('accuracy', 0.0)) if gaze else 0.0,
        }
        _base = _os.path.dirname(_os.path.abspath(__file__))

        # --- Optional ML models for blink / gaze (rule-based fallback) -------
        eye_is_ml = gaze_is_ml = False
        if eye_available:
            _b = ml_modality_score(load_modality_model('blink', _base), blink_features)
            if _b is not None:
                eye_score = _b
                eye_level = _risk_level(_b)
                eye_is_ml = True
        if gaze_available:
            _g = ml_modality_score(load_modality_model('gaze', _base), gaze_features)
            if _g is not None:
                gaze_score = _g
                gaze_level = _risk_level(_g)
                gaze_is_ml = True

        voice_cat = "-"
        voice_pct = "-"
        voice_score_val = None
        if hasattr(self, 'voice_results') and self.voice_results:
            voice_cat = self.voice_results.get('risk_category', '-')
            vp = self.voice_results.get('risk_pct', 0.0)
            if isinstance(vp, (int, float)):
                voice_score_val = float(vp)
                voice_pct = f"{vp:.1f}%"
            else:
                voice_pct = str(vp)

        # --- Overall fused prediction ---------------------------------------
        self.overall_score = compute_overall_score(
            eye_score=eye_score if eye_available else None,
            gaze_score=gaze_score if gaze_available else None,
            voice_score=voice_score_val,
        )
        used_names = {'eye': 'Eye', 'gaze': 'Gaze', 'voice': 'Voice'}
        used_str = " + ".join(used_names[u] for u in self.overall_score.get('used', [])) or "—"

        eye_score_str = (f"{eye_score:.0f}% ({eye_level}){' [ML]' if eye_is_ml else ''}"
                         if eye_available else "N/A (not run)")
        gaze_score_str = (f"{gaze_score:.0f}% ({gaze_level}){' [ML]' if gaze_is_ml else ''}"
                          if gaze_available else "N/A (not run)")
        voice_score_str = f"{voice_pct} ({voice_cat})" if voice_score_val is not None else "N/A (not run)"

        metrics_summary = (
            f"👁  EYE / BLINK   Score: {eye_score_str}   |   "
            f"Blinks: {self.final_metrics.get('total_blinks', 0)}   |   "
            f"Rate: {self.final_metrics.get('blink_rate', 0):.1f} bpm   |   "
            f"Openness: {ear_to_percent(self.final_metrics.get('avg_ear', 0)):.0f}%\n"
            f"🎯  GAZE          Score: {gaze_score_str}   |   "
            f"RT: {avg_rt:.2f}s   |   Saccade: {avg_ss:.0f} px/s   |   "
            f"Accuracy: {acc:.0f}% ({trials} trials)\n"
            f"🎙  VOICE         Score: {voice_score_str}\n"
            f"{'─' * 70}\n"
            f"🧠  OVERALL PREDICTION   {self.overall_score['score']:.0f}%   "
            f"({self.overall_score['level']})   "
            f"[based on: {used_str}]"
        )
        self.final_metrics_text.config(state=tk.NORMAL)
        self.final_metrics_text.delete('1.0', tk.END)
        self.final_metrics_text.insert(tk.END, metrics_summary)
        self.final_metrics_text.config(state=tk.DISABLED)

        # --- Explainable AI: feature contributions to the final score --------
        eye_result_for_xai = None
        if eye_available:
            if eye_is_ml:
                eye_result_for_xai = {'risk_score': eye_score,
                                      'score_breakdown': {'blink_ml_model': eye_score}}
            else:
                eye_result_for_xai = {
                    'risk_score': eye_score,
                    'score_breakdown': self.final_metrics.get('score_breakdown', {}) or {},
                }
        if gaze_available and gaze_is_ml:
            gaze_result_for_xai = {'available': True, 'score': gaze_score,
                                   'breakdown': {'gaze_ml_model': gaze_score}}
        else:
            gaze_result_for_xai = gaze_score_info if gaze_available else None
        voice_result_for_xai = self.voice_results if (hasattr(self, 'voice_results') and self.voice_results) else None

        self.explanation = explain_scores(
            eye_result=eye_result_for_xai,
            gaze_result=gaze_result_for_xai,
            voice_result=voice_result_for_xai,
            overall=self.overall_score,
        )
        self._render_explanation(self.explanation)
        
        # Final EAR Graph
        ear_history = self.final_metrics.get('ear_history', [])
        partial_blinks = self.final_metrics.get('partial_blinks', [])
        self.ax_final1.clear()
        if ear_history:
            left_raw = [ear_to_percent(e['left']) for e in ear_history]
            right_raw = [ear_to_percent(e['right']) for e in ear_history]
            
            # Smooth the final dataset heavily based on length to completely eliminate jitter
            window_size = max(5, len(ear_history) // 30) 
            left_values = smooth_data(left_raw, window_size)
            right_values = smooth_data(right_raw, window_size)
            
            ear_times = [datetime.fromtimestamp(e['timestamp']).strftime('%H:%M:%S') for e in ear_history]
            frames = list(range(len(ear_history)))
            
            self.ax_final1.plot(frames, left_values, color='#3498db', linewidth=1.0, alpha=0.8, label='Left Eye (%)')
            self.ax_final1.plot(frames, right_values, color='#9b59b6', linewidth=1.0, alpha=0.8, label='Right Eye (%)')
            
            thresh_percent = ear_to_percent(self.final_metrics.get('current_threshold', 0.25))
            self.ax_final1.axhline(y=thresh_percent, color='red', linestyle='--', linewidth=1.5, label='Adaptive Blink Threshold')
            
            # Draw Micro-Sleep Warning Zones
            micro_sleeps = self.final_metrics.get('micro_sleeps', [])
            if micro_sleeps:
                for ms in micro_sleeps:
                    start_idx = min(range(len(ear_history)), key=lambda i: abs(ear_history[i]['timestamp'] - ms['start']))
                    end_idx = min(range(len(ear_history)), key=lambda i: abs(ear_history[i]['timestamp'] - ms['end']))
                    if start_idx != end_idx:
                        self.ax_final1.axvspan(start_idx, end_idx, color='red', alpha=0.3, label='Micro-Sleep (>0.5s)' if 'Micro-Sleep (>0.5s)' not in self.ax_final1.get_legend_handles_labels()[1] else "")
            
            if partial_blinks:
                pb_x = []
                pb_y = []
                for pb in partial_blinks:
                    closest_idx = min(range(len(ear_history)), key=lambda i: abs(ear_history[i]['timestamp'] - pb['timestamp']))
                    pb_x.append(closest_idx)
                    pb_y.append(ear_to_percent(pb['min_ear']))
                
                if pb_x:
                    self.ax_final1.scatter(pb_x, pb_y, color='#e67e22', marker='^', s=30, zorder=5, label='Partial Blink')
            
            self.ax_final1.set_title('Final Eye Openness Over Time (L/R Split)', fontdict={'fontsize': 10})
            self.ax_final1.set_ylim(0, 100)
            self.ax_final1.set_ylabel('Eye Openness (%)', fontsize=9)
            self.ax_final1.grid(True, alpha=0.3)
            
            step = max(1, len(frames) // 10)
            self.ax_final1.set_xticks(frames[::step])
            self.ax_final1.set_xticklabels(ear_times[::step], rotation=45, ha='right', fontsize=6)
            self.ax_final1.legend(loc='upper right', fontsize=8)
            self.fig_final1.subplots_adjust(bottom=0.3)
            
        self.canvas_final1.draw()
        
        # Final Blink Graph
        blink_times = self.final_metrics.get('blink_times', [])
        self.ax_final2.clear()
        if len(blink_times) > 1:
            intervals = np.diff(blink_times)
            x_pos = np.arange(len(intervals))
            
            colors = ['#2ecc71' if 2.0 <= interval <= 6.0 else '#e74c3c' for interval in intervals]
            
            y_top = max(5.0, max(intervals) * 1.2)
            # Shade the normal inter-blink interval band (2-6 s)
            self.ax_final2.axhspan(2.0, 6.0, color='#2ecc71', alpha=0.10,
                                   label='Normal interval (2-6s)')
            self.ax_final2.bar(x_pos, intervals, color=colors, width=0.5, label='Interval (s)')
            self.ax_final2.set_title('Final Blink Regularity (Intervals)', fontdict={'fontsize': 10})
            self.ax_final2.set_xlabel('Blink #', fontsize=9)
            self.ax_final2.set_ylabel('Interval (s)', fontsize=9)
            self.ax_final2.set_ylim(0, y_top)
            self.ax_final2.grid(True, alpha=0.3)
            
            labels = [f"Blink {i+1}" for i in range(len(intervals))]
            step = max(1, len(labels) // 10)
            self.ax_final2.set_xticks(x_pos[::step])
            self.ax_final2.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=35, ha='right', fontsize=6)
            self.ax_final2.legend(loc='upper right', fontsize=7)
            self.fig_final2.subplots_adjust(bottom=0.3)
            
        self.canvas_final2.draw()
        
        # Gaze Graph 3: Reaction Time
        self.ax_final3.clear()
        rt_data = gaze.get('reaction_times', [])
        if rt_data:
            x_pos = np.arange(len(rt_data))
            mean_rt = float(np.mean(rt_data))
            # Colour each trial by whether it is within a healthy latency
            bar_colors = ['#2ecc71' if v <= 0.5 else '#f39c12' if v <= 1.0 else '#e74c3c'
                          for v in rt_data]
            self.ax_final3.plot(x_pos, rt_data, color='#f39c12', marker='o',
                                linewidth=2.0, zorder=3, label='Reaction Time')
            self.ax_final3.scatter(x_pos, rt_data, color=bar_colors, s=45, zorder=4)
            # Normal-latency reference band (<= 0.5 s) and mean line
            self.ax_final3.axhspan(0, 0.5, color='#2ecc71', alpha=0.12,
                                   label='Normal (≤0.5s)')
            self.ax_final3.axhline(mean_rt, color='#8e44ad', linestyle='--',
                                   linewidth=1.3, label=f'Mean {mean_rt:.2f}s')
            self.ax_final3.set_title('Gaze Reaction Time per Trial', fontdict={'fontsize': 10})
            self.ax_final3.set_xlabel('Trial', fontsize=9)
            self.ax_final3.set_ylabel('Reaction time (s)', fontsize=9)
            self.ax_final3.set_ylim(0, max(2.0, max(rt_data) * 1.2))
            self.ax_final3.grid(True, alpha=0.3)
            self.ax_final3.set_xticks(x_pos)
            self.ax_final3.set_xticklabels([f"T{i+1}" for i in range(len(rt_data))], fontsize=6)
            self.ax_final3.legend(loc='upper right', fontsize=7)
            self.fig_final3.subplots_adjust(bottom=0.3)
        else:
            self.ax_final3.text(0.5, 0.5, 'Gaze test not run', ha='center', va='center',
                                fontsize=11, color='#a0aec0', transform=self.ax_final3.transAxes)
            self.ax_final3.set_title('Gaze Reaction Time per Trial', fontdict={'fontsize': 10})
        self.canvas_final3.draw()
        
        # Gaze Graph 4: Saccade Speed
        self.ax_final4.clear()
        ss_data = gaze.get('saccade_speeds', [])
        if ss_data:
            x_pos = np.arange(len(ss_data))
            mean_ss = float(np.mean(ss_data))
            # Slow saccades (low speed) are the concern -> colour them red
            bar_colors = ['#e74c3c' if v < 100 else '#f39c12' if v < 300 else '#2ecc71'
                          for v in ss_data]
            self.ax_final4.bar(x_pos, ss_data, color=bar_colors, width=0.5, label='Saccade speed')
            self.ax_final4.axhline(mean_ss, color='#8e44ad', linestyle='--',
                                   linewidth=1.3, label=f'Mean {mean_ss:.0f} px/s')
            self.ax_final4.axhspan(300, max(300, max(ss_data) * 1.2), color='#2ecc71',
                                   alpha=0.10, label='Healthy (≥300 px/s)')
            self.ax_final4.set_title('Saccade Speed per Trial', fontdict={'fontsize': 10})
            self.ax_final4.set_xlabel('Trial', fontsize=9)
            self.ax_final4.set_ylabel('Speed (px/s)', fontsize=9)
            self.ax_final4.set_ylim(0, max(100, max(ss_data) * 1.2))
            self.ax_final4.grid(True, alpha=0.3)
            self.ax_final4.set_xticks(x_pos)
            self.ax_final4.set_xticklabels([f"T{i+1}" for i in range(len(ss_data))], fontsize=6)
            self.ax_final4.legend(loc='upper right', fontsize=7)
            self.fig_final4.subplots_adjust(bottom=0.3)
        else:
            self.ax_final4.text(0.5, 0.5, 'Gaze test not run', ha='center', va='center',
                                fontsize=11, color='#a0aec0', transform=self.ax_final4.transAxes)
            self.ax_final4.set_title('Saccade Speed per Trial', fontdict={'fontsize': 10})
        self.canvas_final4.draw()

    def _render_explanation(self, explanation):
        """Draw the feature-contribution chart and write the plain-language reasons."""
        contribs = explanation.get('contributions', [])
        mod_colors = {'eye': '#3498db', 'gaze': '#f39c12', 'voice': '#9b59b6'}
        mod_label = {'eye': 'Eye/Blink', 'gaze': 'Gaze', 'voice': 'Voice'}

        # --- Bar chart: each feature's contribution toward the final score ---
        self.ax_explain.clear()
        shown = [c for c in contribs if c['contribution'] > 0][:8]
        if shown:
            shown = shown[::-1]  # largest at top in a horizontal bar chart
            labels = [c['label'] for c in shown]
            values = [c['contribution'] for c in shown]
            colors = [mod_colors.get(c['modality'], '#7f8c8d') for c in shown]
            y_pos = range(len(shown))
            self.ax_explain.barh(list(y_pos), values, color=colors)
            self.ax_explain.set_yticks(list(y_pos))
            self.ax_explain.set_yticklabels(labels, fontsize=8)
            for i, v in zip(y_pos, values):
                self.ax_explain.text(v + 0.3, i, f"{v:.1f}", va='center', fontsize=7, color='#333')
            self.ax_explain.set_xlabel('Points contributed to final score', fontsize=8)
            self.ax_explain.set_title('Feature contributions to overall score',
                                      fontsize=10, fontweight='bold', color='#555')
            self.ax_explain.grid(True, axis='x', alpha=0.3)
            # Legend for the modality colours that are actually present
            present = []
            for m in ['eye', 'gaze', 'voice']:
                if any(c['modality'] == m for c in shown):
                    present.append(Patch(facecolor=mod_colors[m], label=mod_label[m]))
            if present:
                self.ax_explain.legend(handles=present, loc='lower right', fontsize=7)
            self.fig_explain.subplots_adjust(left=0.32, bottom=0.18, top=0.88, right=0.95)
        else:
            self.ax_explain.text(0.5, 0.5, 'No feature contributed (all normal)',
                                 ha='center', va='center', fontsize=10, color='#a0aec0',
                                 transform=self.ax_explain.transAxes)
            self.ax_explain.set_xticks([]); self.ax_explain.set_yticks([])
        self.canvas_explain.draw()

        # --- Plain-language explanation text ---------------------------------
        lines = []
        lines.append(explanation.get('summary', ''))
        lines.append("")
        lines.append("How the final score is built:")
        lines.append("overall = sum of (each test's score x its weight).")
        mt = explanation.get('modality_totals', {})
        for m in ['voice', 'eye', 'gaze']:
            if m in mt:
                lines.append(f"  • {mod_label[m]}: +{mt[m]:.1f} points")
        lines.append("")
        lines.append("Top feature drivers (points toward final score):")
        for c in contribs:
            if c['contribution'] > 0:
                lines.append(f"  • {c['label']} (+{c['contribution']:.1f}) — {c['meaning']}")
        if not any(c['contribution'] > 0 for c in contribs):
            lines.append("  • None — all measured features are within normal ranges.")
        lines.append("")
        lines.append("Note: a screening aid, not a diagnosis. Higher score = more "
                     "indicators present, not a confirmed condition.")

        self.explain_text.config(state=tk.NORMAL)
        self.explain_text.delete('1.0', tk.END)
        self.explain_text.insert(tk.END, "\n".join(lines))
        self.explain_text.config(state=tk.DISABLED)

    def generate_report(self):
        """Generate a polished one-page PDF clinical screening report (uses matplotlib)."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception as e:
            messagebox.showerror("Report", f"Could not load PDF backend:\n{e}")
            return
        if not getattr(self, 'final_metrics', None):
            messagebox.showwarning("Report", "Run a detection session first.")
            return

        fm = self.final_metrics
        gaze = getattr(self, 'gaze_results', {}) or {}
        voice = getattr(self, 'voice_results', {}) or {}
        overall = getattr(self, 'overall_score', {}) or {}
        explanation = getattr(self, 'explanation', {}) or {}
        pdata = getattr(self, 'patient_data', {}) or {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        token = str(pdata.get('token', 'NA')).replace(' ', '')
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"report_{token}_{ts}.pdf")

        eye_avail = (fm.get('total_blinks', 0) > 0 or len(fm.get('ear_history', []) or []) > 0)
        gaze_avail = gaze.get('trials', 0) > 0
        voice_avail = bool(voice) and isinstance(voice.get('risk_pct'), (int, float))

        def fmt_score(avail, score, level):
            return f"{score:.0f}%  ({level})" if avail else "Not performed"

        try:
            fig = Figure(figsize=(8.27, 11.69), dpi=120)  # A4 portrait
            fig.subplots_adjust(left=0.07, right=0.93, top=0.97, bottom=0.05)

            # Header band
            fig.text(0.5, 0.955, "Cognitive Screening Report", ha='center',
                     fontsize=20, fontweight='bold', color='#1f4e5f')
            fig.text(0.5, 0.935, "Cognitive Risk Assessment  |  Blink + Gaze + Voice",
                     ha='center', fontsize=10, color='#2e8b8b')
            fig.text(0.5, 0.918, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     ha='center', fontsize=8, color='#777')
            fig.add_artist(plt.Line2D([0.07, 0.93], [0.908, 0.908],
                                      color='#2e8b8b', lw=1.5, transform=fig.transFigure))

            # Patient block
            pinfo = (f"Patient: {pdata.get('name','N/A')}      Token: {pdata.get('token','N/A')}      "
                     f"Age: {pdata.get('age','N/A')}\n"
                     f"Phone: {pdata.get('phone','N/A')}      Place: {pdata.get('place','N/A')}")
            fig.text(0.07, 0.885, pinfo, fontsize=9, va='top', color='#333')

            # Overall score banner
            ov_score = overall.get('score', 0.0)
            ov_level = overall.get('level', 'N/A')
            band_color = {'LOW': '#2ecc71', 'MILD': '#f1c40f', 'MODERATE': '#e67e22',
                          'HIGH': '#e74c3c', 'VERY HIGH': '#c0392b'}.get(ov_level, '#7f8c8d')
            ax_band = fig.add_axes([0.07, 0.80, 0.86, 0.055])
            ax_band.axis('off')
            ax_band.add_patch(plt.Rectangle((0, 0), 1, 1, color=band_color, alpha=0.18))
            ax_band.text(0.02, 0.5, "OVERALL RISK SCORE", va='center', fontsize=11,
                         fontweight='bold', color='#333')
            ax_band.text(0.98, 0.5, f"{ov_score:.0f}%   {ov_level}", va='center', ha='right',
                         fontsize=18, fontweight='bold', color=band_color)

            # Per-test score table
            ax_tbl = fig.add_axes([0.07, 0.64, 0.86, 0.13]); ax_tbl.axis('off')
            rows = [
                ["Test", "Score", "Key measurements"],
                ["Eye / Blink", fmt_score(eye_avail, fm.get('risk_score', 0), fm.get('risk_level', 'N/A')),
                 f"{fm.get('total_blinks',0)} blinks, {fm.get('blink_rate',0):.1f} bpm, "
                 f"{ear_to_percent(fm.get('avg_ear',0)):.0f}% openness"],
                ["Gaze", fmt_score(gaze_avail, compute_gaze_score(gaze).get('score',0),
                                   compute_gaze_score(gaze).get('level','N/A')),
                 f"RT {gaze.get('avg_reaction_time',0):.2f}s, {gaze.get('accuracy',0):.0f}% acc, "
                 f"{gaze.get('trials',0)} trials"],
                ["Voice", (f"{voice.get('risk_pct',0):.0f}%  ({voice.get('risk_category','N/A')})"
                           if voice_avail else "Not performed"),
                 (", ".join(voice.get('key_indicators', [])[:2]) if voice_avail else "-")],
            ]
            tbl = ax_tbl.table(cellText=rows, cellLoc='left', loc='center',
                               colWidths=[0.18, 0.20, 0.62])
            tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor('#dddddd')
                if r == 0:
                    cell.set_facecolor('#2e8b8b'); cell.set_text_props(color='white', fontweight='bold')

            # Explainability chart
            fig.text(0.07, 0.60, "Why this score? — feature contributions to the final score",
                     fontsize=11, fontweight='bold', color='#1f4e5f')
            ax_x = fig.add_axes([0.30, 0.36, 0.60, 0.22])
            contribs = [c for c in explanation.get('contributions', []) if c['contribution'] > 0][:8]
            mod_colors = {'eye': '#3498db', 'gaze': '#f39c12', 'voice': '#9b59b6'}
            if contribs:
                contribs = contribs[::-1]
                ax_x.barh(range(len(contribs)), [c['contribution'] for c in contribs],
                          color=[mod_colors.get(c['modality'], '#888') for c in contribs])
                ax_x.set_yticks(range(len(contribs)))
                ax_x.set_yticklabels([c['label'] for c in contribs], fontsize=7.5)
                ax_x.set_xlabel('Points contributed to final score', fontsize=8)
                for i, c in enumerate(contribs):
                    ax_x.text(c['contribution'] + 0.2, i, f"{c['contribution']:.1f}",
                              va='center', fontsize=6.5)
                ax_x.grid(True, axis='x', alpha=0.3)
                for s in ['top', 'right']:
                    ax_x.spines[s].set_visible(False)
            else:
                ax_x.axis('off')
                ax_x.text(0.5, 0.5, "All measured features within normal ranges",
                          ha='center', va='center', fontsize=10, color='#999')

            # Interpretation text
            lines = [explanation.get('summary', ''), ""]
            mt = explanation.get('modality_totals', {})
            ml = {'eye': 'Eye/Blink', 'gaze': 'Gaze', 'voice': 'Voice'}
            for m in ['voice', 'eye', 'gaze']:
                if m in mt:
                    lines.append(f"  - {ml[m]} contributed +{mt[m]:.1f} points")
            lines.append("")
            lines.append("Top drivers:")
            for c in contribs[::-1][:5] if contribs else []:
                lines.append(f"  - {c['label']} (+{c['contribution']:.1f})")
            fig.text(0.07, 0.335, "\n".join(lines), fontsize=8.5, va='top', color='#333')

            # Footer / disclaimer
            fig.add_artist(plt.Line2D([0.07, 0.93], [0.08, 0.08],
                                      color='#cccccc', lw=1, transform=fig.transFigure))
            disclaimer = ("DISCLAIMER: This is an automated screening aid, NOT a medical diagnosis. "
                          "Scores reflect the presence of behavioural indicators, not a confirmed "
                          "condition. A higher score warrants follow-up with a qualified clinician. "
                          "Voice attribution is an approximate Random-Forest importance estimate.")
            fig.text(0.5, 0.055, disclaimer, ha='center', va='top', fontsize=7,
                     color='#888', wrap=True)

            with PdfPages(out_path) as pdf:
                pdf.savefig(fig)

            messagebox.showinfo("Report Generated",
                                f"PDF report saved to:\n{out_path}")
            try:
                os.startfile(out_path)  # Windows: open the PDF
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Report Error", f"Could not generate report:\n{e}")

    def export_patient_data(self):
        """Export all session and patient data to central CSV"""
        csv_file = "patients_record.csv"
        file_exists = os.path.isfile(csv_file)
        
        # Ordered columns
        headers = [
            "Timestamp", "Token", "Name", "Age", "Phone", "Place", "Address",
            "Total Blinks", "Blink Rate (bpm)", "Avg EAR", "Eye Risk Level", "Eye Score (%)",
            "Avg Reaction Time", "Avg Saccade Speed", "Gaze Accuracy", "Trials", "Gaze Score (%)",
            "Voice Risk Level", "Voice Risk Score (%)", "Speech Rate", "Voice Jitter", "Voice Shimmer", "Voice HNR", "Voice Silence %",
            "Overall Score (%)", "Overall Level", "Top Risk Driver"
        ]
        
        gaze = getattr(self, 'gaze_results', {})
        voice = getattr(self, 'voice_results', {})
        v_met = voice.get('metrics', {}) if isinstance(voice, dict) else {}

        # Per-test and overall scores (recomputed so export is self-contained)
        gaze_info = compute_gaze_score(gaze) if gaze else {'available': False, 'score': ''}
        gaze_score_csv = round(gaze_info['score'], 1) if gaze_info.get('available') else ''
        overall = getattr(self, 'overall_score', None)
        if overall is None:
            eye_s = self.final_metrics.get('risk_score', 0.0)
            v_s = voice.get('risk_pct') if isinstance(voice, dict) else None
            overall = compute_overall_score(
                eye_score=eye_s,
                gaze_score=gaze_info['score'] if gaze_info.get('available') else None,
                voice_score=float(v_s) if isinstance(v_s, (int, float)) else None,
            )
        overall_score_csv = round(overall['score'], 1) if overall.get('available') else ''
        overall_level_csv = overall.get('level', '') if overall.get('available') else ''

        # Top risk driver from the explainability breakdown
        top_driver_csv = ''
        explanation = getattr(self, 'explanation', None)
        if explanation:
            drivers = [c for c in explanation.get('contributions', []) if c['contribution'] > 0]
            if drivers:
                top_driver_csv = f"{drivers[0]['label']} (+{drivers[0]['contribution']:.1f})"

        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.patient_data.get('token', ''),
            self.patient_data.get('name', ''),
            self.patient_data.get('age', ''),
            self.patient_data.get('phone', ''),
            self.patient_data.get('place', ''),
            self.patient_data.get('address', ''),
            self.final_metrics.get('total_blinks', 0),
            round(self.final_metrics.get('blink_rate', 0), 1),
            round(self.final_metrics.get('avg_ear', 0), 3),
            self.final_metrics.get('risk_level', ''),
            round(self.final_metrics.get('risk_score', 0), 0),
            round(gaze.get('avg_reaction_time', 0), 2),
            round(gaze.get('avg_saccade_speed', 0), 2),
            round(gaze.get('accuracy', 0), 1),
            gaze.get('trials', 0),
            gaze_score_csv,
            voice.get('risk_category', '') if voice else '',
            round(voice.get('risk_pct', 0), 1) if voice else '',
            v_met.get('speech_rate', ''),
            v_met.get('jitter', ''),
            v_met.get('shimmer', ''),
            v_met.get('HNR', ''),
            v_met.get('frac_silence', ''),
            overall_score_csv,
            overall_level_csv,
            top_driver_csv
        ]
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow(row_data)
            messagebox.showinfo("Success", f"Data successfully exported to {csv_file}")
            
            # Disable the button after successful export to prevent duplicates
            for widget in self.page3_frame.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Button) and child.cget("text") == "📥 EXPORT TO EXCEL (CSV)":
                            child.config(state=tk.DISABLED)
                            
        except PermissionError:
             messagebox.showerror("Export Error", f"Cannot write to CSV.\n\nThe file '{csv_file}' is currently open in Excel or another program.\nPlease close it and try exporting again.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write to file:\n{str(e)}")
            
    def back_to_detection(self):
        """Go back to Page 2 to run detection again for the same patient"""
        if hasattr(self, 'voice_results'):
            del self.voice_results

        # We need to re-enable export button if it was disabled
        for widget in self.page3_frame.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Button) and child.cget("text") == "📥 EXPORT TO EXCEL (CSV)":
                            child.config(state=tk.NORMAL)
                            
        self.analyzer.reset()
        
        # Reset detection UI state for Page 2
        self.status_label.config(text="System Ready - Click Start to Begin Detection", bg='white', fg='#333')
        self.status_frame.config(bg='white')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.continue_btn.config(state=tk.NORMAL)
        self.auto_stop_enabled = False
        
        # Reset metrics displays
        self.blink_rate_label.config(text="0.0")
        self.total_blinks_label.config(text="0")
        self.eye_openness_label.config(text="0.00")
        self.session_time_label.config(text="0")
        self.risk_label.config(text="Risk Level: LOW", bg='#11998e')
        self.risk_score_label.config(text="0%", bg='#11998e')
        self.risk_frame.config(bg='#11998e')
        self.risk_factors_text.config(state=tk.NORMAL)
        self.risk_factors_text.delete('1.0', tk.END)
        self.risk_factors_text.insert('1.0', 'No risk factors detected')
        self.risk_factors_text.config(state=tk.DISABLED)
        
        # Reset default video frame image
        default_img = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(default_img, "Click Start Detection to begin", (40, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.update_video_frame(default_img)
        
        # Reset live graphs 
        self.ax1.clear()
        self.ax1.set_title('Eye Openness Over Time', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax1.set_ylim(0, 0.5)
        self.ax1.grid(True, alpha=0.3, color='#ccd')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_color('#ccd')
        self.ax1.spines['bottom'].set_color('#ccd')
        self.canvas1.draw()
        
        self.ax2.clear()
        self.ax2.set_title('Blink Pattern', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax2.set_ylim(0, 1.0)
        self.ax2.grid(True, alpha=0.3, color='#ccd')
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_color('#ccd')
        self.ax2.spines['bottom'].set_color('#ccd')
        self.canvas2.draw()
        
        # Explicitly hide page 3 and show page 2
        self.show_page(self.page2_frame)
        
        # Force layout to fully recalculate THEN scroll to top
        self.root.update_idletasks()
        self.root.after(100, lambda: self.canvas.yview_moveto(0))  # increased delay from 50→100
            
    def reset_new_patient(self):
        """Reset forms and metrics for a new patient"""
        self.patient_data = {}
        self.final_metrics = {}
        if hasattr(self, 'voice_results'):
            del self.voice_results
        
        # Reset export button 
        for widget in self.page3_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button) and child.cget("text") == "📥 EXPORT TO EXCEL (CSV)":
                        child.config(state=tk.NORMAL)
                        
        # Clear forms
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            
        self.analyzer.reset()
        
        # Reset detection UI state for Page 2
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.continue_btn.config(state=tk.NORMAL)
        self.auto_stop_enabled = False
        
        # Reset metrics displays
        self.blink_rate_label.config(text="0.0")
        self.total_blinks_label.config(text="0")
        self.eye_openness_label.config(text="0.00")
        self.session_time_label.config(text="0")
        self.risk_label.config(text="Risk Level: LOW", bg='#11998e')
        self.risk_score_label.config(text="0%", bg='#11998e')
        self.risk_frame.config(bg='#11998e')
        self.risk_factors_text.config(state=tk.NORMAL)
        self.risk_factors_text.delete('1.0', tk.END)
        self.risk_factors_text.insert('1.0', 'No risk factors detected')
        self.risk_factors_text.config(state=tk.DISABLED)
        
        # Reset default video frame image
        default_img = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(default_img, "Click Start Detection to begin", (40, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.update_video_frame(default_img)
        
        # Reset live graphs 
        self.ax1.clear()
        self.ax1.set_title('Eye Openness Over Time', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax1.set_ylim(0, 0.5)
        self.ax1.grid(True, alpha=0.3, color='#ccd')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_color('#ccd')
        self.ax1.spines['bottom'].set_color('#ccd')
        self.canvas1.draw()
        
        self.ax2.clear()
        self.ax2.set_title('Blink Pattern', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': '#555'})
        self.ax2.set_ylim(0, 1.0)
        self.ax2.grid(True, alpha=0.3, color='#ccd')
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_color('#ccd')
        self.ax2.spines['bottom'].set_color('#ccd')
        self.canvas2.draw()

        # Explicitly hide page 3 and show page 1
        self.show_page(self.page1_frame)
        
        # Schedule the scroll to top so geometry completes first
        self.root.after(50, lambda: self.canvas.yview_moveto(0))
            
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()


    # ── Voice Analysis ────────────────────────────────────────────────────────
    def start_voice_analysis(self):
        """Launch voice dementia analysis in a background thread."""
        import threading
        import importlib.util
        import os
        import sys

        # Check model files exist
        model_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'dementia_rf_model.pkl')
        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'scaler.pkl')
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            messagebox.showerror(
                "Voice Analysis",
                "Model files not found:\n"
                f"  {model_path}\n  {scaler_path}\n\n"
                "Please ensure dementia_rf_model.pkl and scaler.pkl are "
                "in the project folder.")
            return

        # Build progress dialog with image and terminal
        dlg = tk.Toplevel(self.root)
        dlg.title("🎙 Voice Analysis & Terminal")
        dlg.geometry("900x550")
        dlg.configure(bg='#1a1a2e')
        dlg.resizable(False, False)
        dlg.grab_set()

        main_frame = tk.Frame(dlg, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left frame for image
        left_frame = tk.Frame(main_frame, bg='#1a1a2e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="Cookie Theft Picture", font=("Segoe UI", 12, "bold"), bg='#1a1a2e', fg='white').pack(pady=(0, 10))
        img_lbl = tk.Label(left_frame, bg='#1a1a2e')
        img_lbl.pack()

        try:
            _bdir = os.path.dirname(os.path.abspath(__file__))
            img_path = None
            for _name in ("cookie_theft.jpg", "cookie_theft.png", "cookie_theft.jpeg"):
                _p = os.path.join(_bdir, _name)
                if os.path.exists(_p):
                    img_path = _p
                    break
            if img_path:
                from PIL import Image, ImageTk
                img = Image.open(img_path)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                img_lbl.config(image=photo)
                img_lbl.image = photo
            else:
                img_lbl.config(
                    text="Place 'cookie_theft.jpg' (or .png) in the project\n"
                         "folder to show the picture here.", fg='#e0a000')
        except Exception as e:
            img_lbl.config(text=str(e), fg='red')

        note_lbl = tk.Label(
            left_frame,
            text="⚠ Speak naturally and describe the picture above for 55 seconds after the countdown.",
            font=("Segoe UI", 10), bg='#1a1a2e', fg='#718096',
            wraplength=380)
        note_lbl.pack(pady=10)

        # Right frame for terminal output
        right_frame = tk.Frame(main_frame, bg='#1a1a2e')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        tk.Label(right_frame, text="Terminal Status", font=("Segoe UI", 12, "bold"), bg='#1a1a2e', fg='white').pack(pady=(0, 10))

        txt_frame = tk.Frame(right_frame)
        txt_frame.pack(fill=tk.BOTH, expand=True)
        txt_scroll = tk.Scrollbar(txt_frame)
        txt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        term_text = tk.Text(txt_frame, height=20, width=50, bg='#0d1117', fg='#c9d1d9', font=("Consolas", 9), yscrollcommand=txt_scroll.set)
        term_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt_scroll.config(command=term_text.yview)

        pbar_frame = tk.Frame(right_frame, bg='#2d3748', height=14, highlightthickness=0, bd=0)
        pbar_frame.pack(fill=tk.X, pady=(15, 5))
        pbar_fill = tk.Frame(pbar_frame, bg='#7b2dbe', height=14)
        pbar_fill.place(relwidth=0.0, relheight=1.0)

        cancel_var = [False]
        cancel_btn = tk.Button(
            right_frame, text="Cancel Analysis",
            font=("Segoe UI", 10, "bold"), bg='#e53e3e', fg='white',
            relief=tk.FLAT, cursor='hand2',
            command=lambda: cancel_var.__setitem__(0, True)
        )
        cancel_btn.pack(pady=10)

        class TerminalRedirect:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            def write(self, string):
                def _append():
                    if not self.text_widget.winfo_exists(): return
                    if '\r' in string:
                        parts = string.split('\r')
                        for i, p in enumerate(parts):
                            if i > 0:
                                self.text_widget.delete("end-1c linestart", "end-1c")
                            if p:
                                self.text_widget.insert("end-1c", p)
                    else:
                        self.text_widget.insert(tk.END, string)
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, _append)
            def flush(self):
                pass

        def _update_progress(step, pct):
            try:
                pbar_fill.place(relwidth=pct / 100.0, relheight=1.0)
                dlg.update_idletasks()
            except Exception:
                pass

        def _run():
            # Import voice_dimentia BEFORE overriding sys.stdout, because its global
            # script accesses sys.stdout.buffer on Windows startup.
            try:
                from voice_dimentia import DementiaAnalyser
            except Exception as e:
                # Let the second try block handle the actual failure for GUI display
                pass

            old_stdout = sys.stdout
            sys.stdout = TerminalRedirect(term_text)
            try:
                from voice_dimentia import DementiaAnalyser  # Already loaded
                analyser = DementiaAnalyser(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    indian_accent=True,
                    confidence_threshold=0.60,
                    output_dir=os.path.dirname(os.path.abspath(__file__)))
                result = analyser.run_analysis_for_gui(
                    audio_file=None,
                    plot=True,
                    progress_callback=_update_progress)
            except Exception as exc:
                result = {'error': str(exc), 'prediction': 0,
                          'proba': [1.0, 0.0], 'risk_pct': 0.0,
                          'risk_category': 'Unknown', 'key_indicators': [],
                          'metrics': {}, 'plot_paths': {}}
            finally:
                sys.stdout = old_stdout
            self.root.after(0, lambda: _done(result))

        def _done(result):
            try:
                dlg.destroy()
            except Exception:
                pass
            if cancel_var[0]:
                return
            if result.get('error'):
                messagebox.showerror(
                    "Voice Analysis Error",
                    f"Analysis failed:\n{result['error'][:300]}")
                return
            self.voice_results = result
            VoiceResultsWindow(self.root, result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()


class VoiceResultsWindow(tk.Toplevel):
    """
    Popup dashboard showing the voice dementia analysis results:
      • Key acoustic metrics (speech rate, pitch variability, tremor, HNR)
      • Predicted dementia risk % with colour-coded risk level
      • List of clinical key indicators
      • Buttons to open chart images
    """

    # Risk-level colour palette
    _RISK_COLORS = {
        'Low':      ('#1a472a', '#2ecc71'),   # (bg, fg)
        'Moderate': ('#7b4d00', '#f6ad55'),
        'High':     ('#63171b', '#fc8181'),
        'Unknown':  ('#2d3748', '#a0aec0'),
    }

    def __init__(self, parent, result: dict):
        super().__init__(parent)
        self.title("🎙 Voice Analysis Results")
        self.geometry("620x640")
        self.configure(bg='#1a1a2e')
        self.resizable(True, True)
        self.result = result
        self._build_ui()

    def _build_ui(self):
        r = self.result
        risk_cat  = r.get('risk_category', 'Unknown')
        risk_pct  = r.get('risk_pct', 0.0)
        bg_col, fg_col = self._RISK_COLORS.get(risk_cat,
                                                self._RISK_COLORS['Unknown'])
        metrics   = r.get('metrics', {})
        indicators = r.get('key_indicators', [])
        plot_paths = r.get('plot_paths', {})

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="🎙 Voice Analysis Summary",
                 font=("Segoe UI", 18, "bold"),
                 bg='#1a1a2e', fg='white').pack(pady=(18, 6))

        # ── Risk banner ───────────────────────────────────────────────────────
        risk_frame = tk.Frame(self, bg=bg_col,
                              highlightbackground=fg_col, highlightthickness=2)
        risk_frame.pack(fill=tk.X, padx=30, pady=(4, 12))

        risk_label = ("NO DEMENTIA DETECTED" if r['prediction'] == 0
                      else "DEMENTIA DETECTED")
        tk.Label(risk_frame,
                 text=f"{risk_label}   |   Predicted Risk: {risk_pct:.1f}%   ({risk_cat} Risk)",
                 font=("Segoe UI", 13, "bold"),
                 bg=bg_col, fg=fg_col).pack(pady=10)

        # ── Metrics grid ──────────────────────────────────────────────────────
        metrics_frame = tk.Frame(self, bg='#16213e',
                                 highlightbackground='#2d3748',
                                 highlightthickness=1)
        metrics_frame.pack(fill=tk.X, padx=30, pady=(0, 12))

        tk.Label(metrics_frame, text="Acoustic Metrics",
                 font=("Segoe UI", 11, "bold"),
                 bg='#16213e', fg='#90cdf4').pack(anchor='w', padx=12, pady=(8, 2))

        grid = tk.Frame(metrics_frame, bg='#16213e')
        grid.pack(fill=tk.X, padx=12, pady=(0, 10))

        def _make_metric(parent, row, col, label, value, unit=''):
            cell = tk.Frame(parent, bg='#1e2a45',
                            highlightbackground='#2d3748',
                            highlightthickness=1)
            cell.grid(row=row, column=col, padx=6, pady=5, sticky='nsew')
            tk.Label(cell, text=label, font=("Segoe UI", 9, "bold"),
                     bg='#1e2a45', fg='#90cdf4').pack(pady=(8, 0))
            tk.Label(cell, text=f"{value}",
                     font=("Segoe UI", 17, "bold"),
                     bg='#1e2a45', fg='white').pack()
            tk.Label(cell, text=unit, font=("Segoe UI", 8),
                     bg='#1e2a45', fg='#4a5568').pack(pady=(0, 8))
            parent.grid_columnconfigure(col, weight=1)
            parent.grid_rowconfigure(row, weight=1)

        sr   = metrics.get('speech_rate', 0)
        sr_s = 'Slow' if sr < 2 else ('Normal' if sr <= 4 else 'Fast')
        pv   = metrics.get('pitch_variability', 0)
        jitter   = metrics.get('jitter', 0)
        shimmer  = metrics.get('shimmer', 0)
        hnr      = metrics.get('HNR', 0)
        silence  = metrics.get('frac_silence', 0)

        _make_metric(grid, 0, 0, "Speech Rate",
                     f"{sr:.1f} ({sr_s})", "syllables/sec")
        _make_metric(grid, 0, 1, "Pitch Variability",
                     f"{pv:.1f}", "Hz (std)")
        _make_metric(grid, 0, 2, "Voice Tremor (Jitter)",
                     f"{jitter:.4f}", "ratio")
        _make_metric(grid, 1, 0, "Amplitude Instability",
                     f"{shimmer:.3f}", "(Shimmer)")
        _make_metric(grid, 1, 1, "Voice Clarity (HNR)",
                     f"{hnr:.1f}", "dB")
        _make_metric(grid, 1, 2, "Silence Fraction",
                     f"{silence:.1f}%", "of recording")

        # ── Key Indicators ────────────────────────────────────────────────────
        ind_frame = tk.Frame(self, bg='#16213e',
                             highlightbackground='#2d3748',
                             highlightthickness=1)
        ind_frame.pack(fill=tk.X, padx=30, pady=(0, 12))

        tk.Label(ind_frame, text="Key Indicators Detected",
                 font=("Segoe UI", 11, "bold"),
                 bg='#16213e', fg='#90cdf4').pack(anchor='w', padx=12, pady=(8, 2))

        if indicators:
            for ind in indicators:
                tk.Label(ind_frame,
                         text=f"  ⚠  {ind}",
                         font=("Segoe UI", 10),
                         bg='#16213e', fg='#fc8181',
                         anchor='w').pack(fill=tk.X, padx=12)
        else:
            tk.Label(ind_frame,
                     text="  ✅  No significant warning indicators detected.",
                     font=("Segoe UI", 10),
                     bg='#16213e', fg='#68d391').pack(fill=tk.X, padx=12)
        tk.Label(ind_frame, text="", bg='#16213e').pack(pady=4)

        # ── Disclaimer ────────────────────────────────────────────────────────
        tk.Label(self,
                 text="⚠  Screening tool only — not a medical diagnosis. "
                      "Consult a neurologist for clinical evaluation.",
                 font=("Segoe UI", 8), bg='#1a1a2e', fg='#4a5568',
                 wraplength=560).pack(pady=(0, 8))

        # ── Chart buttons ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg='#1a1a2e')
        btn_frame.pack(pady=(0, 16))

        def _open_image(path):
            try:
                import subprocess, sys
                if sys.platform == 'win32':
                    subprocess.Popen(['explorer', path])
                else:
                    subprocess.Popen(['xdg-open', path])
            except Exception as e:
                messagebox.showerror("Open Image", str(e))

        def _make_btn(parent, text, path, bg):
            import os
            state = tk.NORMAL if (path and os.path.exists(path)) else tk.DISABLED
            tk.Button(parent, text=text,
                      font=("Segoe UI", 10, "bold"),
                      bg=bg, fg='white',
                      relief=tk.FLAT, cursor='hand2',
                      state=state,
                      command=lambda p=path: _open_image(p)
                      ).pack(side=tk.LEFT, padx=6)

        _make_btn(btn_frame, "📊 View Dashboard",
                  plot_paths.get('dashboard'), '#7b2dbe')
        _make_btn(btn_frame, "🌊 View Waveform",
                  plot_paths.get('waveform'), '#2b6cb0')
        _make_btn(btn_frame, "📈 View Features",
                  plot_paths.get('feature'), '#276749')
        tk.Button(btn_frame, text="✖ Close",
                  font=("Segoe UI", 10, "bold"),
                  bg='#4a5568', fg='white',
                  relief=tk.FLAT, cursor='hand2',
                  command=self.destroy).pack(side=tk.LEFT, padx=6)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DementiaDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
