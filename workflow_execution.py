from src.workflow.medical_workflow import MedicalWorkflow
from pathlib import Path

# Initialize workflow
workflow = MedicalWorkflow()

# Define output directory (use absolute path or ensure it exists)
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Run complete workflow
results = workflow.run_complete_workflow(
    audio_path="/home/shahanahmed/neoscoder_task/ElevenLabs_2025-07-07T10_35_21_Rachel_pre_sp100_s50_sb75_v3.mp3",
    selected_diseases=[],  # Auto-generate if empty
    selected_medicines=[],  # Auto-generate if empty
    selected_suggestions=[],  # Auto-generate if empty
    output_dir=output_dir
)

print(f"\n{'='*80}")
print("WORKFLOW COMPLETED SUCCESSFULLY!")
print(f"{'='*80}")
print(f"\nPrescription generated:")
for format_type, filepath in results['exported_files'].items():
    print(f"  - {format_type.upper()}: {filepath}")

print(f"\nPatient: {results['prescription'].patient_name}")
print(f"Diagnoses: {', '.join(results['prescription'].diseases)}")
print(f"Medications prescribed: {len(results['prescription'].medicines)}")
print(f"\n{'='*80}")