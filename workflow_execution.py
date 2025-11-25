from src.workflow.medical_workflow import MedicalWorkflow
from pathlib import Path
import json

def display_menu(title, options):
    """Display a menu and get user selection"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"{'='*80}")


def select_diseases(disease_suggestions):
    """Interactive disease selection"""
    print(f"\n{'='*80}")
    print("🔍 DISEASE SUGGESTIONS FROM AI")
    print(f"{'='*80}")
    
    for i, disease in enumerate(disease_suggestions, 1):
        print(f"\n{i}. {disease['disease']}")
        print(f"   Confidence: {disease['score']:.0%}")
        print(f"   Reason: {disease['reason']}")
    
    print(f"\n{'='*80}")
    print("SELECT DISEASES:")
    print("Enter disease numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 'all' to select all")
    print("Or enter 'custom' to add custom disease")
    print(f"{'='*80}")
    
    selected_diseases = []
    
    while True:
        choice = input("\nYour selection: ").strip().lower()
        
        if choice == 'all':
            selected_diseases = [d['disease'] for d in disease_suggestions]
            break
        elif choice == 'custom':
            custom_disease = input("Enter custom disease name: ").strip()
            if custom_disease:
                selected_diseases.append(custom_disease)
                print(f"✅ Added: {custom_disease}")
            
            cont = input("Add more diseases? (y/n): ").strip().lower()
            if cont != 'y':
                break
        elif choice == 'done':
            break
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 1 <= idx <= len(disease_suggestions):
                        disease_name = disease_suggestions[idx-1]['disease']
                        if disease_name not in selected_diseases:
                            selected_diseases.append(disease_name)
                            print(f"✅ Added: {disease_name}")
                
                cont = input("\nAdd more? (y/n) or 'done' to finish: ").strip().lower()
                if cont != 'y':
                    break
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")
    
    print(f"\n{'='*80}")
    print("SELECTED DISEASES:")
    for disease in selected_diseases:
        print(f"  • {disease}")
    print(f"{'='*80}")
    
    return selected_diseases


def select_medicines(medicine_suggestions):
    """Interactive medicine selection"""
    print(f"\n{'='*80}")
    print("💊 MEDICINE SUGGESTIONS FROM AI")
    print(f"{'='*80}")
    
    for i, med in enumerate(medicine_suggestions, 1):
        print(f"\n{i}. {med['medicine']}")
        print(f"   Score: {med['score']:.0%}")
        print(f"   Reason: {med['reason']}")
        print(f"   Purpose: {med.get('purpose', 'N/A')}")
        print(f"   Side Effects: {med.get('side_effects', 'N/A')}")
    
    print(f"\n{'='*80}")
    print("SELECT MEDICINES:")
    print("Enter medicine numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 'all' to select all")
    print("Or enter 'custom' to add custom medicine")
    print(f"{'='*80}")
    
    selected_medicines = []
    
    while True:
        choice = input("\nYour selection: ").strip().lower()
        
        if choice == 'all':
            selected_medicines = medicine_suggestions.copy()
            break
        elif choice == 'custom':
            custom_medicine = input("Enter custom medicine name: ").strip()
            if custom_medicine:
                # Create basic medicine entry
                custom_med = {
                    "medicine": custom_medicine,
                    "score": 0.0,
                    "reason": "Doctor added",
                    "purpose": "As prescribed",
                    "side_effects": "Consult drug information"
                }
                selected_medicines.append(custom_med)
                print(f"✅ Added: {custom_medicine}")
            
            cont = input("Add more medicines? (y/n): ").strip().lower()
            if cont != 'y':
                break
        elif choice == 'done':
            break
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 1 <= idx <= len(medicine_suggestions):
                        med = medicine_suggestions[idx-1]
                        if med not in selected_medicines:
                            selected_medicines.append(med)
                            print(f"✅ Added: {med['medicine']}")
                
                cont = input("\nAdd more? (y/n) or 'done' to finish: ").strip().lower()
                if cont != 'y':
                    break
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")
    
    print(f"\n{'='*80}")
    print("SELECTED MEDICINES:")
    for med in selected_medicines:
        print(f"  • {med['medicine']}")
    print(f"{'='*80}")
    
    return selected_medicines


def select_suggestions(ai_suggestions, conversation_suggestions):
    """Interactive suggestion selection"""
    print(f"\n{'='*80}")
    print("💡 AI-GENERATED SUGGESTIONS")
    print(f"{'='*80}")
    
    all_suggestions = []
    
    # Display AI suggestions
    for i, suggestion in enumerate(ai_suggestions, 1):
        print(f"{i}. {suggestion}")
        all_suggestions.append(suggestion)
    
    # Display conversation suggestions
    if conversation_suggestions:
        print(f"\n{'='*80}")
        print("💡 SUGGESTIONS FROM CONVERSATION")
        print(f"{'='*80}")
        offset = len(ai_suggestions)
        for i, suggestion in enumerate(conversation_suggestions, 1):
            print(f"{offset + i}. {suggestion}")
            all_suggestions.append(suggestion)
    
    print(f"\n{'='*80}")
    print("SELECT SUGGESTIONS:")
    print("Enter suggestion numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 'all' to select all")
    print("Or enter 'custom' to add custom suggestion")
    print(f"{'='*80}")
    
    selected_suggestions = []
    
    while True:
        choice = input("\nYour selection: ").strip().lower()
        
        if choice == 'all':
            selected_suggestions = all_suggestions.copy()
            break
        elif choice == 'custom':
            custom_suggestion = input("Enter custom suggestion: ").strip()
            if custom_suggestion:
                selected_suggestions.append(custom_suggestion)
                print(f"✅ Added: {custom_suggestion}")
            
            cont = input("Add more suggestions? (y/n): ").strip().lower()
            if cont != 'y':
                break
        elif choice == 'done':
            break
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 1 <= idx <= len(all_suggestions):
                        suggestion = all_suggestions[idx-1]
                        if suggestion not in selected_suggestions:
                            selected_suggestions.append(suggestion)
                            print(f"✅ Added: {suggestion}")
                
                cont = input("\nAdd more? (y/n) or 'done' to finish: ").strip().lower()
                if cont != 'y':
                    break
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")
    
    print(f"\n{'='*80}")
    print("SELECTED SUGGESTIONS:")
    for suggestion in selected_suggestions:
        print(f"  • {suggestion}")
    print(f"{'='*80}")
    
    return selected_suggestions


def save_transcription(transcription, patient_info, output_dir):
    """Save annotated transcription"""
    from datetime import datetime
    
    # Calculate duration
    if transcription:
        timestamps = []
        for item in transcription:
            timestamp_str = item.get('timestamp', '0-0s')
            timestamp_parts = timestamp_str.replace('s', '').split('-')
            if len(timestamp_parts) == 2:
                try:
                    start = float(timestamp_parts[0])
                    end = float(timestamp_parts[1])
                    timestamps.append((start, end))
                except ValueError:
                    continue
        
        if timestamps:
            min_start = min(t[0] for t in timestamps)
            max_end = max(t[1] for t in timestamps)
            duration_minutes = (max_end - min_start) / 60.0
        else:
            duration_minutes = 0
    else:
        duration_minutes = 0
    
    # Create annotated data
    annotated_data = {
        "metadata": {
            "patient_name": patient_info.get('Patient_Name', 'Unknown'),
            "patient_age": patient_info.get('Age', 'N/A'),
            "patient_gender": patient_info.get('Gender', 'N/A'),
            "consultation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_segments": len(transcription),
            "duration_minutes": duration_minutes
        },
        "patient_information": patient_info,
        "conversation": transcription,
        "statistics": {
            "doctor_segments": sum(1 for item in transcription if item['speaker'] == 'Doctor'),
            "patient_segments": sum(1 for item in transcription if item['speaker'] == 'Patient'),
            "total_words": sum(len(item['text'].split()) for item in transcription)
        }
    }
    
    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    patient_name = patient_info.get('Patient_Name', 'patient').replace(' ', '_')
    json_filename = f"annotated_transcription_{patient_name}_{timestamp}.json"
    json_filepath = output_dir / json_filename
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    
    # Save readable text
    txt_filename = f"transcription_{patient_name}_{timestamp}.txt"
    txt_filepath = output_dir / txt_filename
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MEDICAL CONSULTATION TRANSCRIPTION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Patient: {patient_info.get('Patient_Name', 'Unknown')}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration_minutes:.1f} minutes\n\n")
        f.write("="*80 + "\n\n")
        
        for item in transcription:
            speaker_icon = "👨‍⚕️" if item['speaker'] == 'Doctor' else "👤"
            f.write(f"{speaker_icon} {item['speaker']} [{item['timestamp']}]:\n")
            f.write(f"{item['text']}\n\n")
        
        f.write("="*80 + "\n")
    
    return json_filepath, txt_filepath


def main():
    print("\n" + "="*80)
    print("🏥 MEDICAL DIAGNOSTIC SYSTEM - INTERACTIVE MODE")
    print("="*80)
    
    # Initialize workflow
    workflow = MedicalWorkflow()
    
    # Define paths
    audio_path = input("\nEnter audio file path: ").strip()
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process audio
    print("\n🎤 Processing audio file...")
    audio_results = workflow.process_audio_file(audio_path)
    transcription = audio_results['transcription']
    patient_info = audio_results['patient_info']
    
    print(f"✅ Audio processed: {len(transcription)} segments")
    print(f"✅ Patient: {patient_info.get('Patient_Name', 'Unknown')}")
    
    # Save transcription
    print("\n💾 Saving transcription...")
    json_path, txt_path = save_transcription(transcription, patient_info, output_dir)
    print(f"✅ Saved JSON: {json_path}")
    print(f"✅ Saved TXT: {txt_path}")
    
    # Step 2: Generate and select diseases
    print("\n🔍 Generating disease suggestions...")
    disease_suggestions = workflow.generate_diagnoses(patient_info)
    selected_diseases = select_diseases(disease_suggestions)
    
    if not selected_diseases:
        print("⚠️ No diseases selected. Exiting.")
        return
    
    # Step 3: Generate and select medicines
    print("\n💊 Generating medicine suggestions...")
    medicine_suggestions = workflow.generate_medicines(patient_info, selected_diseases)
    selected_medicines = select_medicines(medicine_suggestions)
    
    if not selected_medicines:
        print("⚠️ No medicines selected. Exiting.")
        return
    
    # Step 4: Analyze drug safety
    print("\n⚠️ Analyzing drug interactions...")
    drug_alerts = workflow.analyze_safety(selected_medicines, patient_info)
    
    print(f"\n{'='*80}")
    print(f"DRUG SAFETY ANALYSIS")
    print(f"{'='*80}")
    print(f"Overall Risk Level: {drug_alerts['overall_risk_level']}")
    print(f"Total Alerts: {len(drug_alerts['alerts'])}")
    
    if drug_alerts['alerts']:
        print("\n⚠️ ALERTS:")
        for alert in drug_alerts['alerts']:
            print(f"  • [{alert['severity']}] {alert['description']}")
    
    # Step 5: Generate suggestions
    print("\n💡 Generating suggestions...")
    suggestions_data = workflow.generate_suggestions(
        selected_medicines,
        patient_info,
        selected_diseases
    )
    
    ai_suggestions = suggestions_data['ai_suggestions']
    conversation_suggestions = suggestions_data['conversation_suggestions']
    
    selected_suggestions = select_suggestions(ai_suggestions, conversation_suggestions)
    
    # Step 6: Generate SOAP note
    print("\n📋 Generating SOAP note...")
    soap_note, deidentified_conv = workflow.generate_soap_note(transcription, patient_info)
    print("✅ SOAP note generated")
    
    # Step 7: Generate final prescription
    print("\n📄 Generating final prescription...")
    prescription = workflow.generate_final_prescription(
        patient_info,
        selected_diseases,
        selected_medicines,
        drug_alerts,
        selected_suggestions,
        soap_note
    )
    
    # Step 8: Export prescription
    print("\n💾 Exporting prescription...")
    exported_files = workflow.export_prescription(prescription, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nPatient: {prescription.patient_name}")
    print(f"Diagnoses: {len(selected_diseases)}")
    print(f"Medications: {len(selected_medicines)}")
    print(f"Suggestions: {len(selected_suggestions)}")
    print(f"\nFiles generated:")
    print(f"  • Transcription JSON: {json_path}")
    print(f"  • Transcription TXT: {txt_path}")
    for format_type, filepath in exported_files.items():
        print(f"  • Prescription {format_type.upper()}: {filepath}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()