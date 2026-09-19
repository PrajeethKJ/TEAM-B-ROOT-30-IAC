"""
BioPrint: User Profile Store & Audit Logger
Manages user biometric profiles, enrollment histories, drift states, and audit logs.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "audit_log.json")


class ProfileStore:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
        if not os.path.exists(AUDIT_LOG_FILE):
            with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_profiles(self) -> Dict[str, Any]:
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_profiles(self, profiles: Dict[str, Any]):
        # Save profiles safely (strip non-serializable objects like scikit-learn models from raw JSON)
        serializable = {}
        for user, data in profiles.items():
            user_data = dict(data)
            user_data.pop("_iso_model", None)
            serializable[user] = user_data

        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def save_profile(self, username: str, password_hash: str, biometric_data: Dict[str, Any]):
        profiles = self._load_profiles()
        profiles[username] = {
            "username": username,
            "password_hash": password_hash,
            "created_at": time.time(),
            "biometric_profile": biometric_data
        }
        self._save_profiles(profiles)

    def get_profile(self, username: str) -> Optional[Dict[str, Any]]:
        profiles = self._load_profiles()
        return profiles.get(username)

    def list_profiles(self) -> List[str]:
        profiles = self._load_profiles()
        return list(profiles.keys())

    def update_profile_biometrics(self, username: str, updated_biometrics: Dict[str, Any]):
        profiles = self._load_profiles()
        if username in profiles:
            profiles[username]["biometric_profile"] = updated_biometrics
            profiles[username]["last_updated"] = time.time()
            self._save_profiles(profiles)

    def log_attempt(self, attempt_data: Dict[str, Any]):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

        logs.insert(0, attempt_data)
        # Keep last 50 logs
        logs = logs[:50]

        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def reset_all(self):
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
