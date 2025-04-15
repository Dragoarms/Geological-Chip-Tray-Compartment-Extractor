# /core/translator.py

"""
Handles translations for the application.
Loads translations from CSV files and provides functionality to translate strings.
"""

import os
import csv
import logging
import platform
from typing import Dict, Optional, Any



# TODO: Update FileManager import - from chip_tray_processor.core.file_manager import FileManager

# TODO: Update file path resolution for CSV loading:
# - Check how translations.csv is located 
# - Adjust relative path handling for the new directory structure
# - Consider using a more robust file discovery method

# TODO: Check if __file__ is used for path resolution, which might need updating



class TranslationManager:
    """
    Manages translations for the application.
    Provides functionality to translate strings and detect system language.
    """
    
    def __init__(self, translations_dict=None):
        """
        Initialize the translation manager.
        
        Args:
            translations_dict: Dictionary containing translations for different languages
        """
        # Load translations from CSV file if none provided
        if translations_dict is None:
            self.translations = self._load_translations_from_csv()
        else:
            self.translations = translations_dict
        
        self.current_language = "en"  # Default language
        
        # Try to detect system language
        self.detect_system_language()

    def get_current_language(self):
        """Return the currently selected language."""
        return self.current_language

    def set_language(self, language_code):
        """Set the current language."""
        self.current_language = language_code
    
    def _load_translations_from_csv(self):
        """Load translations from CSV file."""
        translations = {}
        try:
            # Try to load from translations.csv in documents
            csv_data = None
            
            # Get the file path using FileManager
            file_manager = FileManager()
            # Create the Program Resources directory if it doesn't exist
            program_resources_dir = os.path.join(file_manager.base_dir, "Program Resources")
            os.makedirs(program_resources_dir, exist_ok=True)
            
            # Path to translations CSV
            csv_path = os.path.join(program_resources_dir, "translations.csv")
            
            # Check if file exists
            if os.path.exists(csv_path):
                # Read existing file
                csv_encoding = 'cp1252'  # Use the encoding from the document metadata
            else:
                # If file doesn't exist, try to copy from working directory or resources
                # Look for translations.csv in current directory or elsewhere
                possible_paths = [
                    "translations.csv",
                    os.path.join(os.path.dirname(__file__), "translations.csv"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "translations.csv")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        # Copy file to Program Resources directory
                        import shutil
                        shutil.copy2(path, csv_path)
                        csv_encoding = 'cp1252'
                        break
            
            # Now try to read the CSV file
            if os.path.exists(csv_path):
                import csv
                with open(csv_path, 'r', encoding=csv_encoding) as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Get column headers
                    
                    # First column is the key, others are language codes
                    key_index = 0
                    language_indices = {}
                    
                    # Extract the language codes from headers
                    for i, header in enumerate(headers):
                        if i > 0:  # Skip the first column (key)
                            language_indices[header] = i
                            translations[header] = {}
                    
                    # Read each row and populate translations
                    for row in reader:
                        if len(row) > 0:
                            key = row[key_index]
                            for lang, index in language_indices.items():
                                if index < len(row) and row[index]:  # Only add non-empty translations
                                    translations[lang][key] = row[index]
            
            return translations
            
        except Exception as e:
            print(f"Error loading translations: {e}")
            # Return empty dictionary if there was an error
            return {}
    
    def detect_system_language(self):
        """Detect system language from locale settings."""
        try:
            import locale
            # Get system platform
            system = platform.system()

            if system == 'Windows':
                # Set and get the current locale
                locale.setlocale(locale.LC_ALL, '')
                language_code = locale.getlocale()
                if language_code:  # language_code is a tuple, (language, encoding)
                    # Extract primary language
                    lang = language_code[0].split('_')[0].lower() if language_code[0] else None
                    if lang and lang in self.translations:
                        self.current_language = lang
                        return True
            elif system in ('Darwin', 'Linux'):
                # Try to get locale from environment variables
                lang_env = os.environ.get('LANG', '')
                if lang_env:
                    lang = lang_env.split('_')[0].lower()
                    if lang in self.translations:
                        self.current_language = lang
                        return True
        except Exception as e:
            print(f"Error detecting system language: {e}")

        return False
    
    def set_language(self, language_code):
        """
        Set the current language.
        
        Args:
            language_code: Language code (e.g., "en", "fr")
            
        Returns:
            bool: True if language was set successfully, False otherwise
        """
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_available_languages(self):
        """
        Get list of available languages.
        
        Returns:
            List of language codes
        """
        return list(self.translations.keys())
    
    def get_language_name(self, language_code):
        """Get human-readable name for a language code."""
        language_names = {
            "en": "English",
            "fr": "Français",
            "american": "American English"
            # Add more language names as needed
        }
        return language_names.get(language_code, language_code)
    
    def translate(self, text):
        """
        Translate text to the current language.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text or original text if no translation found
        """
        # Handle None or empty string
        if text is None or text == "":
            return text
            
        # Get translation dictionary for current language
        translations = self.translations.get(self.current_language, {})
        
        # Return translation if available, otherwise return original text
        return translations.get(text, text)

    def t(self, text):
        """Shorthand method for translate."""
        return self.translate(text)
    
