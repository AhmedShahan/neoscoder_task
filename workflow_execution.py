from src.workflow.medical_workflow import MedicalWorkflow
from pathlib import Path

workflow = MedicalWorkflow()

# Run complete workflow
results = workflow.run_complete_workflow(
    audio_path="/home/shahanahmed/neoscoder_task/ElevenLabs_2025-07-07T10_35_21_Rachel_pre_sp100_s50_sb75_v3.mp3",
    selected_diseases=[],  # Auto-generate if empty
    selected_medicines=[],  # Auto-generate if empty
    selected_suggestions=[],  # Auto-generate if empty
    output_dir=Path("./outputs")
)

print(f"Prescription generated: {results['exported_files']}")