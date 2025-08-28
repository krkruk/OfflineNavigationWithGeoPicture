import pandas as pd
import logging

class LandmarkManager:
    """Handles loading and managing landmark data from a CSV file."""
    def __init__(self, landmarks_file):
        self.landmarks_file = landmarks_file
        self.landmarks = self._load_landmarks()

    def _load_landmarks(self):
        """Loads landmarks from the CSV file."""
        try:
            landmarks = pd.read_csv(self.landmarks_file, index_col=0)
            # Requirement 2: use only capital letters
            landmarks.index = landmarks.index.str.upper()
            logging.info(f"Successfully loaded landmarks from {self.landmarks_file}")
            return landmarks
        except FileNotFoundError:
            logging.error(f"Landmark file not found: {self.landmarks_file}")
            return None
        except Exception as e:
            logging.error(f"Error loading landmarks: {e}")
            return None
