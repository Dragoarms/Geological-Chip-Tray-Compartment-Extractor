# gui/update_dialog.py

"""
Dialog for checking and applying updates to the application.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import threading
import time
import logging
from typing import Optional, Callable

# Import DialogHelper
from .dialog_helper import DialogHelper

# Make sure the core directory is in the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import RepoUpdater
from core.repo_updater import RepoUpdater

logger = logging.getLogger(__name__)

class UpdateDialog:
    """
    Dialog for checking and applying updates to the application.
    """
    
    def __init__(self, parent, config_manager=None, callback=None):
        """
        Initialize the update dialog.
        
        Args:
            parent: Parent window
            config_manager: Configuration manager instance
            callback: Function to call after update completes
        """
        self.parent = parent
        self.config_manager = config_manager
        self.callback = callback
        self.dialog = None
        self.updater = None
        self.check_thread = None
        self.status_var = tk.StringVar(value="Initializing...")
        self.version_var = tk.StringVar(value="")
        
        # Create the updater
        self._create_updater()
        
        # Create the dialog
        self._create_dialog()
    
    def _create_updater(self):
        """Create the repository updater."""
        try:
            # Get GitHub repository URL from config if available
            github_repo = "https://github.com/Dragoarms/Geological-Chip-Tray-Compartment-Extractor"
            branch = "main"
            
            if self.config_manager:
                if self.config_manager.get("github_repo"):
                    github_repo = self.config_manager.get("github_repo")
                if self.config_manager.get("github_branch"):
                    branch = self.config_manager.get("github_branch")
            
            # Create the updater
            self.updater = RepoUpdater(
                config_manager=self.config_manager,
                github_repo=github_repo,
                branch=branch,
                dialog_helper=DialogHelper
            )
            
        except Exception as e:
            logger.error(f"Error creating updater: {e}")
            self.status_var.set(f"Error: {str(e)}")
    
    def _create_dialog(self):
        """Create the update dialog."""
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            "Check for Updates",
            modal=True,
            topmost=True,
            size_ratio=0.4,
            min_width=450,
            min_height=250
        )
        
        # Create main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            header_frame,
            text="Update Chip Tray Processor",
            font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT)
        
        # Create version info
        version_frame = ttk.Frame(main_frame)
        version_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            version_frame,
            text="Current Version:"
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            version_frame,
            textvariable=self.version_var
        ).pack(side=tk.RIGHT)
        
        # Create status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=400
        )
        self.status_label.pack(fill=tk.X)
        
        # Create progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=15)
        
        # Create buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.check_button = ttk.Button(
            button_frame,
            text="Check for Updates",
            command=self._check_for_updates
        )
        self.check_button.pack(side=tk.LEFT)
        
        self.update_button = ttk.Button(
            button_frame,
            text="Update Now",
            command=self._apply_update,
            state=tk.DISABLED
        )
        self.update_button.pack(side=tk.LEFT, padx=10)
        
        self.close_button = ttk.Button(
            button_frame,
            text="Close",
            command=self._close_dialog
        )
        self.close_button.pack(side=tk.RIGHT)
        
        # Set up event handling
        self.dialog.protocol("WM_DELETE_WINDOW", self._close_dialog)
        self.dialog.bind("<Escape>", lambda e: self._close_dialog())
        
        # Start checking for updates automatically
        if self.updater:
            self.version_var.set(self.updater.get_local_version())
            self.dialog.after(500, self._check_for_updates)
    
    def _check_for_updates(self):
        """Check for updates in a separate thread."""
        if self.check_thread and self.check_thread.is_alive():
            # Already checking
            return
        
        # Disable the check button
        self.check_button.config(state=tk.DISABLED)
        self.update_button.config(state=tk.DISABLED)
        
        # Set status
        self.status_var.set("Checking for updates...")
        self.progress_var.set(0)
        
        # Start checking in a separate thread
        self.check_thread = threading.Thread(target=self._check_thread_func)
        self.check_thread.daemon = True
        self.check_thread.start()
        
        # Schedule update of UI
        self._update_ui_from_thread()
    
    def _check_thread_func(self):
        """Thread function for checking updates."""
        try:
            # Compare versions
            result = self.updater.compare_versions()
            
            if result['error']:
                self.status_var.set(f"Error checking for updates: {result['error']}")
                return
            
            if result['update_available']:
                self.status_var.set(
                    f"A new version is available: {result['github_version']}\n"
                    f"Your current version: {result['local_version']}\n\n"
                    "Click 'Update Now' to download and install the update."
                )
                # Enable the update button
                self.dialog.after(0, lambda: self.update_button.config(state=tk.NORMAL))
            else:
                self.status_var.set(
                    f"You have the latest version: {result['local_version']}\n\n"
                    "No updates are available at this time."
                )
        except Exception as e:
            logger.error(f"Error in check thread: {e}")
            self.status_var.set(f"Error checking for updates: {str(e)}")
    
    def _update_ui_from_thread(self):
        """Update UI based on thread status."""
        if self.check_thread and self.check_thread.is_alive():
            # Thread is still running, check again in 100ms
            self.dialog.after(100, self._update_ui_from_thread)
        else:
            # Thread completed, re-enable the check button
            self.check_button.config(state=tk.NORMAL)
    
    def _apply_update(self):
        """Apply the update."""
        # Disable buttons
        self.check_button.config(state=tk.DISABLED)
        self.update_button.config(state=tk.DISABLED)
        self.close_button.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Downloading and applying updates...\nThis may take a few minutes.")
        
        # Create a progress dialog
        self.progress_dialog = None
        
        # Start update in a separate thread
        update_thread = threading.Thread(target=self._update_thread_func)
        update_thread.daemon = True
        update_thread.start()
        
        # Schedule update of UI
        self._monitor_update_progress(update_thread)
    
    def _update_thread_func(self):
        """Thread function for applying updates."""
        try:
            # Execute the update
            result = self.updater.execute_update(self.dialog)
            
            # Update status based on result
            if result['errors'] == 0:
                self.status_var.set(
                    f"Update completed successfully!\n\n"
                    f"- {result['updated_files']} files updated\n"
                    f"- {result['new_files']} new files created\n"
                    f"- {result['unchanged_files']} files unchanged\n\n"
                    "Restart the application to apply changes."
                )
                
                # Add restart button
                self.dialog.after(0, self._add_restart_button)
            else:
                self.status_var.set(
                    f"Update completed with errors!\n\n"
                    f"- {result['updated_files']} files updated\n"
                    f"- {result['new_files']} new files created\n"
                    f"- {result['unchanged_files']} files unchanged\n"
                    f"- {result['errors']} errors occurred\n\n"
                    "Some files could not be updated. The application may not function correctly."
                )
        except Exception as e:
            logger.error(f"Error in update thread: {e}")
            self.status_var.set(f"Error applying updates: {str(e)}")
    
    def _monitor_update_progress(self, thread):
        """Monitor update progress."""
        if thread.is_alive():
            # Thread is still running, check again in 100ms
            self.dialog.after(100, lambda: self._monitor_update_progress(thread))
        else:
            # Thread completed, re-enable close button
            self.close_button.config(state=tk.NORMAL)
            
            # Update progress bar to 100%
            self.progress_var.set(100)
    
    def _add_restart_button(self):
        """Add a restart button to the dialog."""
        # Create restart button next to close button
        button_frame = self.close_button.master
        
        restart_button = ttk.Button(
            button_frame,
            text="Restart Now",
            command=self._restart_application
        )
        restart_button.pack(side=tk.RIGHT, padx=(0, 10))
    
    def _restart_application(self):
        """Restart the application."""
        try:
            # Close the dialog
            self._close_dialog()
            
            # Restart the application
            if self.updater:
                self.updater.restart_application()
        except Exception as e:
            logger.error(f"Error restarting application: {e}")
            DialogHelper.show_message(
                self.parent,
                "Restart Failed",
                f"Failed to restart the application: {str(e)}\n\n"
                "Please close and restart the application manually.",
                message_type="error"
            )
    
    def _close_dialog(self):
        """Close the dialog."""
        if self.check_thread and self.check_thread.is_alive():
            # Wait for check thread to complete
            self.check_thread.join(0.5)
        
        # Destroy the dialog
        if self.dialog:
            self.dialog.destroy()
            self.dialog = None
        
        # Call the callback if provided
        if self.callback:
            self.callback()
    
    @staticmethod
    def show(parent, config_manager=None, callback=None):
        """
        Show the update dialog.
        
        Args:
            parent: Parent window
            config_manager: Configuration manager instance
            callback: Function to call after dialog closes
        """
        dialog = UpdateDialog(parent, config_manager, callback)
        return dialog.dialog
