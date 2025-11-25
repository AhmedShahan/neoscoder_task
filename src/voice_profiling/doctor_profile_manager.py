"""
Doctor Profile Manager Module
Manages doctor voice profiles and embeddings
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from .embedding_generator import EmbeddingGenerator
from ..config import MODEL_PATHS


class DoctorProfileManager:
    """
    Manages doctor voice profiles and their embeddings
    """
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize doctor profile manager
        
        Args:
            profiles_dir: Directory to store doctor profiles
        """
        self.profiles_dir = profiles_dir or Path(MODEL_PATHS['DOCTOR_EMBEDDING_PATH']).parent
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles_file = self.profiles_dir / "doctor_profiles.json"
        self.embedding_generator = EmbeddingGenerator()
        
        # Load existing profiles
        self.profiles = self._load_profiles()
    
    def _load_profiles(self) -> Dict:
        """Load existing doctor profiles"""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_profiles(self):
        """Save doctor profiles to file"""
        with open(self.profiles_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def create_profile(
        self,
        doctor_name: str,
        audio_path: str,
        doctor_info: Optional[Dict] = None
    ) -> str:
        """
        Create a new doctor profile from audio
        
        Args:
            doctor_name: Name of the doctor
            audio_path: Path to audio file containing doctor's voice
            doctor_info: Optional dictionary with doctor information
            
        Returns:
            Profile ID
        """
        print(f"Creating profile for Dr. {doctor_name}...")
        
        # Generate unique profile ID
        profile_id = doctor_name.lower().replace(' ', '_')
        
        # Generate embedding
        print("Generating voice embedding...")
        embedding = self.embedding_generator.generate_embedding_from_file(audio_path)
        
        # Save embedding
        embedding_filename = f"{profile_id}_embedding.npy"
        embedding_path = self.profiles_dir / embedding_filename
        self.embedding_generator.save_embedding(embedding, embedding_path)
        
        # Create profile
        profile = {
            "doctor_name": doctor_name,
            "profile_id": profile_id,
            "embedding_path": str(embedding_path),
            "created_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "audio_source": audio_path,
            "doctor_info": doctor_info or {}
        }
        
        # Save profile
        self.profiles[profile_id] = profile
        self._save_profiles()
        
        print(f"✅ Profile created successfully!")
        print(f"   Profile ID: {profile_id}")
        print(f"   Embedding saved: {embedding_path}")
        
        return profile_id
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """
        Get doctor profile by ID
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Profile dictionary or None
        """
        return self.profiles.get(profile_id)
    
    def get_embedding(self, profile_id: str) -> Optional[np.ndarray]:
        """
        Get doctor embedding by profile ID
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Embedding array or None
        """
        profile = self.get_profile(profile_id)
        if profile:
            embedding_path = profile['embedding_path']
            return self.embedding_generator.load_embedding(embedding_path)
        return None
    
    def list_profiles(self) -> List[Dict]:
        """
        List all doctor profiles
        
        Returns:
            List of profile dictionaries
        """
        return list(self.profiles.values())
    
    def update_profile(
        self,
        profile_id: str,
        audio_path: Optional[str] = None,
        doctor_info: Optional[Dict] = None
    ) -> bool:
        """
        Update existing doctor profile
        
        Args:
            profile_id: Profile identifier
            audio_path: Optional new audio file
            doctor_info: Optional updated doctor information
            
        Returns:
            True if successful, False otherwise
        """
        profile = self.get_profile(profile_id)
        if not profile:
            print(f"❌ Profile '{profile_id}' not found")
            return False
        
        # Update embedding if new audio provided
        if audio_path:
            print("Updating voice embedding...")
            embedding = self.embedding_generator.generate_embedding_from_file(audio_path)
            embedding_path = profile['embedding_path']
            self.embedding_generator.save_embedding(embedding, embedding_path)
            profile['audio_source'] = audio_path
            profile['updated_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update doctor info
        if doctor_info:
            profile['doctor_info'].update(doctor_info)
        
        self.profiles[profile_id] = profile
        self._save_profiles()
        
        print(f"✅ Profile '{profile_id}' updated successfully!")
        return True
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete doctor profile
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            True if successful, False otherwise
        """
        profile = self.get_profile(profile_id)
        if not profile:
            print(f"❌ Profile '{profile_id}' not found")
            return False
        
        # Delete embedding file
        embedding_path = Path(profile['embedding_path'])
        if embedding_path.exists():
            embedding_path.unlink()
        
        # Remove from profiles
        del self.profiles[profile_id]
        self._save_profiles()
        
        print(f"✅ Profile '{profile_id}' deleted successfully!")
        return True
    
    def set_active_profile(self, profile_id: str) -> bool:
        """
        Set active doctor profile for speaker identification
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            True if successful, False otherwise
        """
        profile = self.get_profile(profile_id)
        if not profile:
            print(f"❌ Profile '{profile_id}' not found")
            return False
        
        # Copy embedding to default location
        embedding = self.get_embedding(profile_id)
        default_path = Path(MODEL_PATHS['DOCTOR_EMBEDDING_PATH'])
        self.embedding_generator.save_embedding(embedding, default_path)
        
        print(f"✅ Active profile set to: {profile['doctor_name']}")
        print(f"   Embedding copied to: {default_path}")
        return True
    
    def display_profiles(self):
        """Display all doctor profiles"""
        if not self.profiles:
            print("No doctor profiles found.")
            return
        
        print(f"\n{'='*80}")
        print("DOCTOR PROFILES")
        print(f"{'='*80}")
        
        for profile_id, profile in self.profiles.items():
            print(f"\nProfile ID: {profile_id}")
            print(f"Doctor Name: {profile['doctor_name']}")
            print(f"Created: {profile['created_date']}")
            print(f"Embedding: {profile['embedding_path']}")
            
            if profile.get('doctor_info'):
                print("Additional Info:")
                for key, value in profile['doctor_info'].items():
                    print(f"  {key}: {value}")
        
        print(f"{'='*80}\n")


# Convenience functions
def create_doctor_profile(doctor_name: str, audio_path: str, doctor_info: Optional[Dict] = None) -> str:
    """
    Create a doctor profile
    
    Args:
        doctor_name: Name of the doctor
        audio_path: Path to audio file
        doctor_info: Optional doctor information
        
    Returns:
        Profile ID
    """
    manager = DoctorProfileManager()
    return manager.create_profile(doctor_name, audio_path, doctor_info)


def set_active_doctor(profile_id: str) -> bool:
    """
    Set active doctor for speaker identification
    
    Args:
        profile_id: Profile identifier
        
    Returns:
        True if successful
    """
    manager = DoctorProfileManager()
    return manager.set_active_profile(profile_id)