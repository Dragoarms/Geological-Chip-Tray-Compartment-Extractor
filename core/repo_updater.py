# core/repo_updater.py

"""
Enhanced updater for downloading and updating the entire codebase from GitHub.
Maintains directory structure and handles multiple files.
"""

import os
import sys
import ssl
import json
import re
import shutil
import logging
import tempfile
import hashlib
import urllib.request
import time
import subprocess
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import queue

# Configure logging
logger = logging.getLogger(__name__)

class RepoUpdater:
    """
    Advanced updater for downloading and updating the entire application codebase.
    Preserves directory structure and handles incremental updates.
    """
    
    def __init__(self, 
                config_manager=None,
                github_repo="https://github.com/Dragoarms/Geological-Chip-Tray-Compartment-Extractor",
                branch="main",
                token=None,
                dialog_helper=None):
        """
        Initialize the repository updater.
        
        Args:
            config_manager: Configuration manager instance
            github_repo: URL to the GitHub repository
            branch: Branch to download from
            token: GitHub personal access token for private repositories
            dialog_helper: DialogHelper instance for user interactions
        """
        self.github_repo = github_repo.rstrip(".git")  # Remove .git if present
        self.branch = branch
        self.token = token
        self.logger = logging.getLogger(__name__)
        self.dialog_helper = dialog_helper
        self.config_manager = config_manager
        self.progress_queue = queue.Queue()
        
        # Extract owner and repo name from the URL
        match = re.search(r'github\.com/([^/]+)/([^/.]+)', github_repo)
        if match:
            self.owner = match.group(1)
            self.repo = match.group(2)
        else:
            self.owner = None
            self.repo = None
        
        # Statistics for reporting
        self.stats = {
            'total_files': 0,
            'updated_files': 0,
            'new_files': 0,
            'unchanged_files': 0,
            'errors': 0,
        }
        
        # Get the scripts directory from config or use default
        self.scripts_dir = self._get_scripts_directory()
        
        # File type patterns to include in updates
        self.include_patterns = [
            r'\.py$',           # Python files
            r'\.json$',         # JSON files
            r'\.csv$',          # CSV files
            r'\.md$',           # Markdown files
            r'README',          # README files
            r'LICENSE',         # License files
            r'requirements\.txt$', # Python requirements
        ]
        
        # Files and directories to exclude from updates
        self.exclude_patterns = [
            r'__pycache__',     # Python cache
            r'\.git',           # Git directory
            r'\.vscode',        # VSCode settings
            r'\.idea',          # PyCharm settings
            r'user_config\.json$', # User-specific config
            r'local_settings\.py$', # Local settings
            r'output/',         # Output directory
            r'venv/',           # Virtual environment
            r'\.env$',          # Environment variables
        ]
    
    def _get_scripts_directory(self) -> str:
        """
        Get the scripts directory from configuration or use a default.
        
        Returns:
            str: Path to the scripts directory
        """
        # Try to get from config manager
        if self.config_manager:
            scripts_dir = self.config_manager.get("scripts_directory")
            if scripts_dir:
                return scripts_dir
        
        # Try to read from config.json directly
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "scripts_directory" in config:
                        return config["scripts_directory"]
        except Exception as e:
            self.logger.warning(f"Error reading scripts_directory from config.json: {e}")
        
        # Default to current directory's parent
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def get_local_version(self) -> str:
        """
        Get the local version of the application.
        
        Returns:
            str: Local version as a string
        """
        # Try reading from __init__.py
        try:
            init_path = os.path.join(self.scripts_dir, "__init__.py")
            if os.path.exists(init_path):
                with open(init_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
        except Exception as e:
            self.logger.warning(f"Error reading version from __init__.py: {e}")
        
        # Try to load from the main module
        try:
            # Check for __version__ in main module
            import __main__
            if hasattr(__main__, '__version__'):
                return __main__.__version__
        except Exception:
            pass
        
        # Try reading from config.json
        try:
            config_path = os.path.join(self.scripts_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "version" in config:
                        return config["version"]
        except Exception as e:
            self.logger.warning(f"Error reading version from config.json: {e}")
        
        # Fallback to hardcoded version
        return "1.0"  # Default fallback version
    
    def get_github_version(self) -> str:
        """
        Get the latest version from GitHub.
        
        Returns:
            str: Latest version as a string, or "Unknown" if not found
        """
        try:
            if not self.owner or not self.repo:
                return "Unknown"
                
            # Possible file paths where version might be stored
            possible_paths = [
                f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/__init__.py",
                f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/version.txt",
                f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/config.json",
            ]
            
            context = ssl._create_unverified_context()
            
            # Try each path
            for raw_url in possible_paths:
                try:
                    # Create a request object
                    request = urllib.request.Request(raw_url)
                    
                    # Add authorization header if token is provided
                    if self.token:
                        request.add_header("Authorization", f"token {self.token}")
                    
                    # Make the request
                    response = urllib.request.urlopen(request, context=context)
                    content = response.read().decode('utf-8')
                    
                    if raw_url.endswith("__init__.py"):
                        # Parse version from Python file
                        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                        if match:
                            return match.group(1)
                    elif raw_url.endswith("version.txt"):
                        # Parse version from text file
                        match = re.search(r'[Vv]ersion\s*=\s*["\']*(\d+\.\d+(?:\.\d+)?)["\']*', content)
                        if match:
                            return match.group(1)
                    elif raw_url.endswith("config.json"):
                        # Parse version from JSON
                        try:
                            config = json.loads(content)
                            if "version" in config:
                                return config["version"]
                        except json.JSONDecodeError:
                            pass
                except Exception as path_error:
                    self.logger.warning(f"Failed with path {raw_url}: {str(path_error)}")
                    continue
            
            # If all paths fail, return "Unknown"
            return "Unknown"
        except Exception as e:
            self.logger.error(f"Error getting GitHub version: {str(e)}")
            return "Unknown"
    
    def compare_versions(self) -> dict:
        """
        Compare local and GitHub versions.

        Returns:
            Dictionary with comparison results
        """
        local_version = self.get_local_version()
        github_version = self.get_github_version()

        result = {
            'local_version': local_version,
            'github_version': github_version,
            'update_available': False,
            'error': None
        }

        if local_version == "Unknown" or github_version == "Unknown":
            result['error'] = "Could not determine versions"
            return result

        try:
            # Convert to tuples of integers for comparison
            local_parts = tuple(map(int, local_version.split('.')))
            github_parts = tuple(map(int, github_version.split('.')))

            # Pad with zeros if versions have different number of parts
            max_length = max(len(local_parts), len(github_parts))
            local_parts = local_parts + (0,) * (max_length - len(local_parts))
            github_parts = github_parts + (0,) * (max_length - len(github_parts))

            result['update_available'] = github_parts > local_parts

            return result
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error comparing versions: {str(e)}")
            return result
    
    def _is_file_included(self, file_path: str) -> bool:
        """
        Check if a file should be included in the update.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file should be included, False otherwise
        """
        # Check against exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return False
        
        # Then check against include patterns
        for pattern in self.include_patterns:
            if re.search(pattern, file_path):
                return True
        
        # Default to excluding files that don't match any pattern
        return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate the MD5 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: MD5 hash of the file content
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _download_file(self, url: str, target_path: str) -> bool:
        """
        Download a file from URL to target path.
        
        Args:
            url: URL to download from
            target_path: Path to save the file to
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Create a request object
            request = urllib.request.Request(url)
            
            # Add authorization header if token is provided
            if self.token:
                request.add_header("Authorization", f"token {self.token}")
            
            # Create SSL context
            context = ssl._create_unverified_context()
            
            # Download to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with urllib.request.urlopen(request, context=context) as response:
                    shutil.copyfileobj(response, tmp_file)
            
            # Move the temporary file to the target location
            shutil.move(tmp_path, target_path)
            
            self.logger.info(f"Downloaded {url} to {target_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False
    
    def _get_repo_contents(self, path: str = "") -> List[Dict[str, Any]]:
        """
        Get the contents of a repository directory from GitHub API.
        
        Args:
            path: Path within the repository (empty for root)
            
        Returns:
            List of dictionaries with file/directory information
        """
        try:
            # Create API URL
            api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}?ref={self.branch}"
            
            # Create request
            request = urllib.request.Request(api_url)
            request.add_header("Accept", "application/vnd.github.v3+json")
            
            # Add authorization header if token is provided
            if self.token:
                request.add_header("Authorization", f"token {self.token}")
            
            # Create SSL context
            context = ssl._create_unverified_context()
            
            # Make the request
            with urllib.request.urlopen(request, context=context) as response:
                content = response.read().decode('utf-8')
                return json.loads(content)
        except Exception as e:
            self.logger.error(f"Error getting repository contents for {path}: {e}")
            return []
    
    def _scan_repo_recursively(self, path: str = "", files: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Recursively scan the repository for files.
        
        Args:
            path: Current path within the repository
            files: List to append files to
            
        Returns:
            List of dictionaries with file information
        """
        if files is None:
            files = []
        
        contents = self._get_repo_contents(path)
        
        for item in contents:
            item_path = item['path']
            
            # Check exclusions
            if not any(re.search(pattern, item_path) for pattern in self.exclude_patterns):
                if item['type'] == 'file':
                    # Check if this file should be included
                    if self._is_file_included(item_path):
                        files.append(item)
                elif item['type'] == 'dir':
                    # Recursively scan directories
                    self._scan_repo_recursively(item_path, files)
        
        return files
    
    def _create_update_plan(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a plan for updating files.
        
        Returns:
            Dictionary with lists of files to update, create, and ignore
        """
        self.logger.info("Creating update plan...")
        
        # Get all files from the repository
        repo_files = self._scan_repo_recursively()
        
        # Categorize files
        update_plan = {
            'update': [],    # Files that exist locally but are different
            'create': [],    # Files that don't exist locally
            'ignore': []     # Files that are the same or should be skipped
        }
        
        for file_info in repo_files:
            file_path = file_info['path']
            local_path = os.path.join(self.scripts_dir, file_path)
            
            # Skip excluded files
            if not self._is_file_included(file_path):
                update_plan['ignore'].append(file_info)
                continue
            
            # Check if file exists locally
            if os.path.exists(local_path):
                # Compare file content or size/date to determine if update is needed
                # For simplicity, we'll just check if the file sizes are different
                # In a real implementation, you might want to compare file hashes or timestamps
                try:
                    # Download the file to a temporary location to compare
                    tmp_fd, tmp_path = tempfile.mkstemp()
                    os.close(tmp_fd)
                    
                    if self._download_file(file_info['download_url'], tmp_path):
                        # Compare the temporary file with the local file
                        if self._get_file_hash(tmp_path) != self._get_file_hash(local_path):
                            update_plan['update'].append(file_info)
                        else:
                            update_plan['ignore'].append(file_info)
                    
                    # Remove the temporary file
                    os.remove(tmp_path)
                except Exception as e:
                    self.logger.error(f"Error comparing files for {file_path}: {e}")
                    # Default to updating the file if comparison fails
                    update_plan['update'].append(file_info)
            else:
                # File doesn't exist locally, so create it
                update_plan['create'].append(file_info)
        
        self.logger.info(f"Update plan: {len(update_plan['update'])} files to update, "
                        f"{len(update_plan['create'])} files to create, "
                        f"{len(update_plan['ignore'])} files to ignore")
        
        return update_plan
    
    def execute_update(self, parent_window=None) -> Dict[str, Any]:
        """
        Execute the update plan.
        
        Args:
            parent_window: Optional parent window for progress dialog
            
        Returns:
            Dictionary with update statistics
        """
        self.logger.info("Starting update process...")
        
        # Reset statistics
        self.stats = {
            'total_files': 0,
            'updated_files': 0,
            'new_files': 0,
            'unchanged_files': 0,
            'errors': 0,
        }
        
        # Create update plan
        update_plan = self._create_update_plan()
        
        # Calculate total files
        self.stats['total_files'] = len(update_plan['update']) + len(update_plan['create'])
        
        # Create a progress dialog if parent window is provided
        progress_dialog = None
        progress_var = None
        
        if parent_window and self.stats['total_files'] > 0:
            progress_dialog = tk.Toplevel(parent_window)
            progress_dialog.title("Updating Application")
            progress_dialog.geometry("400x150")
            progress_dialog.transient(parent_window)
            progress_dialog.grab_set()
            
            # Create progress bar and labels
            ttk.Label(progress_dialog, text="Downloading and updating files...").pack(pady=10)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            status_label = ttk.Label(progress_dialog, text="Preparing...")
            status_label.pack(pady=10)
            
            # Update progress dialog
            def update_progress():
                if progress_dialog.winfo_exists():
                    # Check for progress updates
                    try:
                        while not self.progress_queue.empty():
                            progress_info = self.progress_queue.get_nowait()
                            if 'progress' in progress_info:
                                progress_var.set(progress_info['progress'])
                            if 'status' in progress_info:
                                status_label.config(text=progress_info['status'])
                        
                        # Schedule next update
                        progress_dialog.after(100, update_progress)
                    except queue.Empty:
                        progress_dialog.after(100, update_progress)
            
            # Start progress updates
            progress_dialog.after(100, update_progress)
            progress_dialog.update()
        
        # Process files to update
        completed_files = 0
        for file_info in update_plan['update']:
            file_path = file_info['path']
            local_path = os.path.join(self.scripts_dir, file_path)
            
            # Update progress
            completed_files += 1
            progress = (completed_files / self.stats['total_files']) * 100 if self.stats['total_files'] > 0 else 100
            
            if progress_dialog:
                self.progress_queue.put({
                    'progress': progress,
                    'status': f"Updating {file_path}..."
                })
                progress_dialog.update()
            
            # Download and update the file
            if self._download_file(file_info['download_url'], local_path):
                self.stats['updated_files'] += 1
                self.logger.info(f"Updated {file_path}")
            else:
                self.stats['errors'] += 1
                self.logger.error(f"Failed to update {file_path}")
        
        # Process files to create
        for file_info in update_plan['create']:
            file_path = file_info['path']
            local_path = os.path.join(self.scripts_dir, file_path)
            
            # Update progress
            completed_files += 1
            progress = (completed_files / self.stats['total_files']) * 100 if self.stats['total_files'] > 0 else 100
            
            if progress_dialog:
                self.progress_queue.put({
                    'progress': progress,
                    'status': f"Creating {file_path}..."
                })
                progress_dialog.update()
            
            # Download and create the file
            if self._download_file(file_info['download_url'], local_path):
                self.stats['new_files'] += 1
                self.logger.info(f"Created {file_path}")
            else:
                self.stats['errors'] += 1
                self.logger.error(f"Failed to create {file_path}")
        
        # Count unchanged files
        self.stats['unchanged_files'] = len(update_plan['ignore'])
        
        # Complete the progress
        if progress_dialog:
            self.progress_queue.put({
                'progress': 100,
                'status': "Update completed!"
            })
            progress_dialog.update()
            
            # Close the dialog after a short delay
            progress_dialog.after(1500, progress_dialog.destroy)
        
        self.logger.info(f"Update completed: {self.stats['updated_files']} files updated, "
                        f"{self.stats['new_files']} files created, "
                        f"{self.stats['unchanged_files']} files unchanged, "
                        f"{self.stats['errors']} errors")
        
        return self.stats
    
    def check_and_update(self, parent_window=None) -> Dict[str, Any]:
        """
        Check for updates and apply them if available.
        
        Args:
            parent_window: Optional parent window for dialogs
            
        Returns:
            Dictionary with update result
        """
        # Check if update is available
        version_check = self.compare_versions()
        
        if version_check['error']:
            self.logger.warning(f"Error checking for updates: {version_check['error']}")
            
            # Show error dialog if dialog helper is available
            if self.dialog_helper and parent_window:
                self.dialog_helper.show_message(
                    parent_window,
                    "Update Check Failed",
                    f"Could not check for updates: {version_check['error']}",
                    message_type="error"
                )
            
            return {'success': False, 'message': version_check['error']}
        
        if not version_check['update_available']:
            self.logger.info("No updates available.")
            
            # Show info dialog if dialog helper is available
            if self.dialog_helper and parent_window:
                self.dialog_helper.show_message(
                    parent_window,
                    "No Updates Available",
                    f"You have the latest version: {version_check['local_version']}",
                    message_type="info"
                )
            
            return {'success': True, 'message': "No updates available", 'updated': False}
        
        # Update is available - ask for confirmation
        should_update = True
        
        if self.dialog_helper and parent_window:
            should_update = self.dialog_helper.confirm_dialog(
                parent_window,
                "Update Available",
                f"A new version is available: {version_check['github_version']}\n"
                f"Your current version: {version_check['local_version']}\n\n"
                "Do you want to update now?"
            )
        
        if should_update:
            # Execute the update
            update_stats = self.execute_update(parent_window)
            
            # Show update results
            update_message = (
                f"Update completed successfully!\n\n"
                f"- {update_stats['updated_files']} files updated\n"
                f"- {update_stats['new_files']} new files created\n"
                f"- {update_stats['unchanged_files']} files unchanged\n"
            )
            
            if update_stats['errors'] > 0:
                update_message += f"- {update_stats['errors']} errors occurred\n\n"
                update_message += "Some errors occurred during the update. The application may not function correctly."
            
            if self.dialog_helper and parent_window:
                message_type = "info" if update_stats['errors'] == 0 else "warning"
                self.dialog_helper.show_message(
                    parent_window,
                    "Update Completed",
                    update_message,
                    message_type=message_type
                )
            
            return {
                'success': update_stats['errors'] == 0,
                'message': update_message,
                'updated': True,
                'stats': update_stats
            }
        else:
            return {'success': True, 'message': "Update was canceled", 'updated': False}
    
    def restart_application(self):
        """Restart the application after update."""
        try:
            # Get path to the main script
            main_script = os.path.join(self.scripts_dir, "main.py")
            if os.path.exists(main_script):
                self.logger.info(f"Restarting application from {main_script}")
                
                # Start a new process
                subprocess.Popen([sys.executable, main_script])
                
                # Exit the current process
                self.logger.info("Exiting current process")
                sys.exit(0)
            else:
                self.logger.error(f"Main script not found at {main_script}")
        except Exception as e:
            self.logger.error(f"Error restarting application: {e}")
