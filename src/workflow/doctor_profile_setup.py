"""
Doctor Profile Setup CLI
Interactive command-line tool to manage doctor profiles
"""
from src.voice_profiling.doctor_profile_manager import DoctorProfileManager
# from src.voice.doctor_profile_manager import DoctorProfileManager
from pathlib import Path


def display_menu():
    """Display main menu"""
    print("\n" + "="*80)
    print("👨‍⚕️ DOCTOR PROFILE MANAGEMENT")
    print("="*80)
    print("1. Create new doctor profile")
    print("2. List all profiles")
    print("3. Set active profile")
    print("4. Update profile")
    print("5. Delete profile")
    print("6. Exit")
    print("="*80)


def create_profile_interactive(manager):
    """Interactive profile creation"""
    print("\n" + "="*80)
    print("CREATE NEW DOCTOR PROFILE")
    print("="*80)
    
    # Get doctor information
    doctor_name = input("Enter doctor's full name: ").strip()
    if not doctor_name:
        print("❌ Doctor name is required!")
        return
    
    audio_path = input("Enter path to doctor's voice audio file: ").strip()
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Optional information
    print("\nOptional Information (press Enter to skip):")
    specialty = input("Specialty: ").strip()
    license_number = input("License Number: ").strip()
    hospital = input("Hospital/Clinic: ").strip()
    
    doctor_info = {}
    if specialty:
        doctor_info['specialty'] = specialty
    if license_number:
        doctor_info['license_number'] = license_number
    if hospital:
        doctor_info['hospital'] = hospital
    
    # Create profile
    try:
        profile_id = manager.create_profile(doctor_name, audio_path, doctor_info)
        
        # Ask if this should be the active profile
        set_active = input("\nSet this as active profile? (y/n): ").strip().lower()
        if set_active == 'y':
            manager.set_active_profile(profile_id)
        
    except Exception as e:
        print(f"❌ Error creating profile: {str(e)}")


def list_profiles_interactive(manager):
    """List all profiles"""
    manager.display_profiles()
    
    if manager.profiles:
        print("\nPress Enter to continue...")
        input()


def set_active_profile_interactive(manager):
    """Set active profile"""
    if not manager.profiles:
        print("\n❌ No profiles found. Create a profile first.")
        return
    
    print("\n" + "="*80)
    print("SET ACTIVE PROFILE")
    print("="*80)
    
    # List profiles
    profiles = list(manager.profiles.keys())
    for i, profile_id in enumerate(profiles, 1):
        profile = manager.profiles[profile_id]
        print(f"{i}. {profile['doctor_name']} (ID: {profile_id})")
    
    # Get selection
    try:
        choice = int(input("\nSelect profile number: ").strip())
        if 1 <= choice <= len(profiles):
            profile_id = profiles[choice - 1]
            manager.set_active_profile(profile_id)
        else:
            print("❌ Invalid selection!")
    except ValueError:
        print("❌ Invalid input!")


def update_profile_interactive(manager):
    """Update existing profile"""
    if not manager.profiles:
        print("\n❌ No profiles found. Create a profile first.")
        return
    
    print("\n" + "="*80)
    print("UPDATE PROFILE")
    print("="*80)
    
    # List profiles
    profiles = list(manager.profiles.keys())
    for i, profile_id in enumerate(profiles, 1):
        profile = manager.profiles[profile_id]
        print(f"{i}. {profile['doctor_name']} (ID: {profile_id})")
    
    # Get selection
    try:
        choice = int(input("\nSelect profile number: ").strip())
        if 1 <= choice <= len(profiles):
            profile_id = profiles[choice - 1]
            
            # Get updates
            print("\nLeave blank to keep current value:")
            audio_path = input("New audio file path: ").strip()
            
            specialty = input("Specialty: ").strip()
            license_number = input("License Number: ").strip()
            hospital = input("Hospital/Clinic: ").strip()
            
            doctor_info = {}
            if specialty:
                doctor_info['specialty'] = specialty
            if license_number:
                doctor_info['license_number'] = license_number
            if hospital:
                doctor_info['hospital'] = hospital
            
            # Update
            audio_path = audio_path if audio_path else None
            doctor_info = doctor_info if doctor_info else None
            
            manager.update_profile(profile_id, audio_path, doctor_info)
        else:
            print("❌ Invalid selection!")
    except ValueError:
        print("❌ Invalid input!")


def delete_profile_interactive(manager):
    """Delete profile"""
    if not manager.profiles:
        print("\n❌ No profiles found.")
        return
    
    print("\n" + "="*80)
    print("DELETE PROFILE")
    print("="*80)
    
    # List profiles
    profiles = list(manager.profiles.keys())
    for i, profile_id in enumerate(profiles, 1):
        profile = manager.profiles[profile_id]
        print(f"{i}. {profile['doctor_name']} (ID: {profile_id})")
    
    # Get selection
    try:
        choice = int(input("\nSelect profile number to delete: ").strip())
        if 1 <= choice <= len(profiles):
            profile_id = profiles[choice - 1]
            
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete '{profile_id}'? (y/n): ").strip().lower()
            if confirm == 'y':
                manager.delete_profile(profile_id)
        else:
            print("❌ Invalid selection!")
    except ValueError:
        print("❌ Invalid input!")


def main():
    """Main function"""
    manager = DoctorProfileManager()
    
    while True:
        display_menu()
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            create_profile_interactive(manager)
        elif choice == '2':
            list_profiles_interactive(manager)
        elif choice == '3':
            set_active_profile_interactive(manager)
        elif choice == '4':
            update_profile_interactive(manager)
        elif choice == '5':
            delete_profile_interactive(manager)
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option! Please select 1-6.")


if __name__ == "__main__":
    main()