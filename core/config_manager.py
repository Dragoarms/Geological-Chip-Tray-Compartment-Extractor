# core/config_manager.py

import os
import json

class ConfigManager:
    """A simple configuration manager for loading and saving JSON configuration files.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self._save()

    def _save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def as_dict(self):
        return self.config
