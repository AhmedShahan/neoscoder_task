import streamlit as st
import os
import requests
import re
import json
import time
import logging
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from dotenv import load_dotenv
import pandas as pd
from io import StringIO

# Load environment
load_dotenv()
# from langchain_ollama import ChatOllama
model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5
)

# Setup logging
def setup_logging():
    """Setup logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('patient_system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration Management
class AppConfig:
    def __init__(self):
        self.api_timeout = 10
        self.max_retries = 3
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.cache_duration = 3600  # 1 hour
        self.load_from_env()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        self.api_timeout = int(os.getenv("API_TIMEOUT", "10"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))

config = AppConfig()

# Debug: Print API key to verify
if config.debug_mode:
    print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
    print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

# Provided conversation list
try:
    from demo_conversation import recorded_list2
    record_list = recorded_list2
except ImportError:
    logger.warning("demo_conversation module not found, using sample data")
    record_list = [
        ("Doctor", "Hello, how are you feeling today?"),
        ("Patient", "I've been having some headaches lately.")
    ]

class PatientInformation(BaseModel):
    Patient_Name: Optional[str] = Field(default=None, description="Name of the patient")
    Age: Optional[int] = Field(default=None, description="Age of the patient")
    Gender: Optional[Literal["Male", "Female", "Other"]] = Field(default=None, description="Gender of the patient")
    Recent_Problem: Optional[str] = Field(default=None, description="Primary health issue reported by the patient")
    Previous_Medical_History: Optional[str] = Field(default=None, description="Past medical conditions or surgeries")
    Previous_Drug_History: Optional[str] = Field(default=None, description="Medications previously or currently used")
    Allergies: Optional[str] = Field(default=None, description="Known allergies of the patient")
    Family_Medical_History: Optional[str] = Field(default=None, description="Medical conditions in the patient's family")
    Lifestyle_Details: Optional[Dict[str, str]] = Field(default=None, description="Details of the patient's lifestyle")
    Current_Medical_Tests_Ordered: Optional[str] = Field(default=None, description="Medical tests ordered during the visit")
    Previous_Medical_Test_History: Optional[str] = Field(default=None, description="Previous medical tests conducted")
    Follow_Up_Actions: Optional[List[str]] = Field(default=None, description="List of follow-up actions or recommendations")
    Emotional_State: Optional[str] = Field(default=None, description="Emotional state of the patient during the visit")
    Dialogue: Optional[List[Dict[str, str]]] = Field(default=None, description="List of dialogue lines with speaker and text")

class DrugNames(BaseModel):
    drug_names: List[str] = Field(description="List of clean drug names extracted from drug history, containing only the medication names without dosages, frequencies, or additional information")

class DrugInteraction(BaseModel):
    drug1: str = Field(description="First drug name")
    drug2: str = Field(description="Second drug name")
    interaction_level: str = Field(description="Interaction severity: Minor, Moderate, Major, or None")
    description: str = Field(description="Description of the interaction")

def validate_patient_data(patient_data: Dict) -> Dict:
    """Validate and sanitize patient data"""
    validated_data = patient_data.copy()
    
    # Validate age
    if validated_data.get("Age"):
        age = validated_data["Age"]
        if not isinstance(age, int) or age < 0 or age > 150:
            st.warning("⚠️ Invalid age detected, please verify patient information")
            logger.warning(f"Invalid age detected: {age}")
    
    # Validate gender
    valid_genders = ["Male", "Female", "Other"]
    if validated_data.get("Gender") and validated_data["Gender"] not in valid_genders:
        validated_data["Gender"] = "Other"
        logger.info(f"Gender normalized to 'Other'")
    
    # Sanitize text fields
    text_fields = ["Recent_Problem", "Previous_Medical_History", "Allergies", "Family_Medical_History"]
    for field in text_fields:
        if validated_data.get(field):
            validated_data[field] = validated_data[field].strip()
    
    return validated_data

def extract_drug_names(drug_history: str) -> List[str]:
    """Extract clean drug names from drug history text using LLM"""
    if not drug_history:
        return []
    
    try:
        # Initialize the model
        llm = model
        
        # Define prompt template for drug name extraction
        drug_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical drug name extractor. Your task is to extract ONLY the drug names from the given drug history text.
            
            Rules:
            1. Extract only the actual medication names (generic or brand names)
            2. Remove all dosages (e.g., 500mg, 50mcg, 100mg)
            3. Remove all frequencies (e.g., daily, twice daily, as needed, bid, tid, qid)
            4. Remove all additional information (e.g., "for headaches", "used twice")
            5. Remove all parenthetical information
            6. Return only clean drug names
            
            Examples:
            Input: "Metformin 500mg daily, Levothyroxine 50 mcg daily"
            Output: ["Metformin", "Levothyroxine"]
            
            Input: "Ibuprofen 400mg as needed for headaches, Aspirin 81mg daily"
            Output: ["Ibuprofen", "Aspirin"]"""),
            ("human", "Extract drug names from this drug history: {drug_history}\n\n{format_instructions}")
        ])
        
        # Create parser
        drug_parser = PydanticOutputParser(pydantic_object=DrugNames)
        
        # Create the chain
        chain = drug_extraction_prompt | llm | drug_parser
        
        # Invoke the chain
        result = chain.invoke({
            "drug_history": drug_history,
            "format_instructions": drug_parser.get_format_instructions()
        })
        
        logger.info(f"Extracted drug names: {result.drug_names}")
        return result.drug_names
        
    except Exception as e:
        st.error(f"Error extracting drug names: {str(e)}")
        logger.error(f"Error extracting drug names: {str(e)}")
        # Fallback to regex-based extraction
        return extract_drug_names_fallback(drug_history)

def extract_drug_names_fallback(drug_history: str) -> List[str]:
    """Fallback method to extract drug names using regex"""
    if not drug_history:
        return []
    
    logger.info("Using fallback method for drug name extraction")
    
    # Split by commas and clean each drug
    drugs = []
    for drug in drug_history.split(","):
        # Remove dosage patterns (numbers followed by units)
        cleaned_drug = re.sub(r'\s*\d+\s*(mg|g|mcg|ml|units?)\b', '', drug.strip(), flags=re.IGNORECASE)
        # Remove frequency patterns
        cleaned_drug = re.sub(r'\s*(daily|twice daily|as needed|bid|tid|qid|once|twice|thrice)\b', '', cleaned_drug, flags=re.IGNORECASE)
        # Remove parenthetical information
        cleaned_drug = re.sub(r'\s*\([^)]*\)', '', cleaned_drug)
        # Remove "for" conditions
        cleaned_drug = cleaned_drug.split(" for ")[0].strip()
        # Remove "used" information
        cleaned_drug = re.sub(r'\s*used\s+.*', '', cleaned_drug, flags=re.IGNORECASE)
        
        if cleaned_drug:
            drugs.append(cleaned_drug.strip())
    
    return drugs

def get_drug_purpose_with_retry(drug_name: str, max_retries: int = None) -> Dict:
    """Enhanced drug lookup with retry logic and alternative search methods"""
    if max_retries is None:
        max_retries = config.max_retries
    
    for attempt in range(max_retries):
        try:
            # Try brand name first
            url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
            response = requests.get(url, timeout=config.api_timeout)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    logger.info(f"Found drug info for {drug_name} via brand name search")
                    return data['results'][0]
            
            # Try generic name if brand name fails
            url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
            response = requests.get(url, timeout=config.api_timeout)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    logger.info(f"Found drug info for {drug_name} via generic name search")
                    return data['results'][0]
                    
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {drug_name}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": f"Network error after {max_retries} attempts: {str(e)}"}
            time.sleep(1)  # Wait before retry
    
    return {"error": "No results found for the drug after trying multiple search methods"}

def check_drug_interactions(drug_names: List[str]) -> List[Dict]:
    """Check for potential drug interactions (placeholder implementation)"""
    # This is a placeholder - in production, this would integrate with a drug interaction API
    interactions = []
    
    # Common drug interactions (simplified example)
    interaction_pairs = {
        ("Warfarin", "Aspirin"): {"level": "Major", "description": "Increased bleeding risk"},
        ("Metformin", "Alcohol"): {"level": "Moderate", "description": "Risk of lactic acidosis"},
        ("Lisinopril", "Potassium"): {"level": "Moderate", "description": "Hyperkalemia risk"},
    }
    
    for i, drug1 in enumerate(drug_names):
        for drug2 in drug_names[i+1:]:
            pair = (drug1, drug2)
            reverse_pair = (drug2, drug1)
            
            if pair in interaction_pairs:
                interaction = interaction_pairs[pair]
                interactions.append({
                    "drug1": drug1,
                    "drug2": drug2,
                    "interaction_level": interaction["level"],
                    "description": interaction["description"]
                })
            elif reverse_pair in interaction_pairs:
                interaction = interaction_pairs[reverse_pair]
                interactions.append({
                    "drug1": drug2,
                    "drug2": drug1,
                    "interaction_level": interaction["level"],
                    "description": interaction["description"]
                })
    
    return interactions

def search_patient_history(patient_data: Dict, search_term: str) -> List[str]:
    """Search through patient history for specific terms"""
    results = []
    searchable_fields = ["Previous_Medical_History", "Previous_Drug_History", "Allergies", "Family_Medical_History"]
    
    for field in searchable_fields:
        if patient_data.get(field) and search_term.lower() in patient_data[field].lower():
            results.append(f"Found in {field}: {patient_data[field]}")
    
    return results

def export_patient_data(patient_data: Dict, format_type: str = "json") -> str:
    """Export patient data in various formats"""
    if format_type == "json":
        return json.dumps(patient_data, indent=2)
    elif format_type == "csv":
        # Convert to CSV format
        df = pd.DataFrame([patient_data])
        return df.to_csv(index=False)
    elif format_type == "text":
        # Convert to readable text format
        text_output = []
        for key, value in patient_data.items():
            if value:
                text_output.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(text_output)
    return ""

def extract_patient_info(conversation):
    """Extract patient information from conversation and pre-fetch drug details"""
    try:
        # Initialize the model with the same settings as the working code
        llm = model
        
        # Define prompt template
        prescription_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Patient Information Extractor. Extract relevant patient information from the conversation and respond with valid JSON matching the provided format. Ensure the output is a valid JSON string and contains only the fields specified in the format instructions."),
            ("human", "{conversation}\n\n{format_instructions}")
        ])
        
        patient_parser = PydanticOutputParser(pydantic_object=PatientInformation)
        
        # Convert conversation list of tuples to string
        conversation_text = "\n".join([f"{speaker}: {text}" for speaker, text in conversation])
        
        # Create the chain
        chain = prescription_prompt | llm | patient_parser
        
        # Invoke the chain
        result = chain.invoke({
            "conversation": conversation_text,
            "format_instructions": patient_parser.get_format_instructions()
        })
        
        # Ensure Dialogue field is populated
        if result.Dialogue is None:
            result.Dialogue = [{"speaker": speaker, "text": text} for speaker, text in conversation]
        
        raw_data = result.dict()
        validated_data = validate_patient_data(raw_data)
        
        # Pre-fetch drug details if Previous_Drug_History exists
        if validated_data.get("Previous_Drug_History"):
            drug_names = extract_drug_names(validated_data["Previous_Drug_History"])
            if drug_names:
                # Initialize drug cache if not already present
                if "drug_cache" not in st.session_state:
                    st.session_state["drug_cache"] = {}
                
                # Fetch details for each drug and store in cache
                for drug in drug_names:
                    if drug not in st.session_state["drug_cache"]:
                        drug_data = get_drug_purpose_with_retry(drug)
                        st.session_state["drug_cache"][drug] = drug_data
                        logger.info(f"Cached drug details for {drug}")
        
        logger.info("Patient information extracted successfully")
        return validated_data
        
    except Exception as e:
        st.error(f"Error processing conversation: {str(e)}")
        logger.error(f"Error processing conversation: {str(e)}")
        return None

def display_basic_info(patient_data):
    """Display basic patient information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if patient_data.get("Patient_Name"):
            st.metric("Patient Name", patient_data["Patient_Name"])
    
    with col2:
        if patient_data.get("Age"):
            st.metric("Age", f"{patient_data['Age']} years")
    
    with col3:
        if patient_data.get("Gender"):
            st.metric("Gender", patient_data["Gender"])
    
    if patient_data.get("Recent_Problem"):
        st.subheader("🩺 Current Health Issue")
        st.write(patient_data["Recent_Problem"])
    
    if patient_data.get("Emotional_State"):
        st.subheader("😊 Emotional State")
        st.info(patient_data["Emotional_State"])

def display_medications_tab(patient_data):
    """Display medications with interaction checking"""
    if patient_data.get("Previous_Drug_History"):
        st.subheader("💊 Current Medications")
        st.write(patient_data["Previous_Drug_History"])
        
        # Extract clean drug names
        drug_names = extract_drug_names(patient_data["Previous_Drug_History"])
        
        if drug_names:
            st.subheader("📋 Extracted Medications")
            for drug in drug_names:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {drug}")
                with col2:
                    if st.button(f"Details", key=f"details_{drug}"):
                        st.session_state["current_drug"] = drug
                        st.session_state["page"] = "drug_details"
            
            # Check for drug interactions
            st.subheader("⚠️ Drug Interactions")
            interactions = check_drug_interactions(drug_names)
            
            if interactions:
                for interaction in interactions:
                    level_color = {
                        "Major": "error",
                        "Moderate": "warning", 
                        "Minor": "info"
                    }.get(interaction["interaction_level"], "info")
                    
                    if level_color == "error":
                        st.error(f"**{interaction['interaction_level']}** - {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")
                    elif level_color == "warning":
                        st.warning(f"**{interaction['interaction_level']}** - {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")
                    else:
                        st.info(f"**{interaction['interaction_level']}** - {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")
            else:
                st.success("No known drug interactions detected")
    
    if patient_data.get("Allergies"):
        st.subheader("⚠️ Allergies")
        st.error(patient_data["Allergies"])

def display_tests_tab(patient_data):
    """Display tests and results"""
    col1, col2 = st.columns(2)
    
    with col1:
        if patient_data.get("Current_Medical_Tests_Ordered"):
            st.subheader("🧪 Tests Ordered")
            st.info(patient_data["Current_Medical_Tests_Ordered"])
    
    with col2:
        if patient_data.get("Previous_Medical_Test_History"):
            st.subheader("📊 Previous Tests")
            st.write(patient_data["Previous_Medical_Test_History"])
    
    if patient_data.get("Follow_Up_Actions"):
        st.subheader("📝 Follow-up Actions")
        for action in patient_data["Follow_Up_Actions"]:
            st.write(f"• {action}")

def display_history_tab(patient_data):
    """Display medical history"""
    col1, col2 = st.columns(2)
    
    with col1:
        if patient_data.get("Previous_Medical_History"):
            st.subheader("📋 Medical History")
            st.write(patient_data["Previous_Medical_History"])
    
    with col2:
        if patient_data.get("Family_Medical_History"):
            st.subheader("👨‍👩‍👧‍👦 Family History")
            st.write(patient_data["Family_Medical_History"])
    
    if patient_data.get("Lifestyle_Details"):
        st.subheader("🏃‍♂️ Lifestyle Information")
        lifestyle = patient_data["Lifestyle_Details"]
        
        if isinstance(lifestyle, dict):
            cols = st.columns(min(len(lifestyle), 4))
            for i, (key, value) in enumerate(lifestyle.items()):
                with cols[i % 4]:
                    st.write(f"**{key.replace('_', ' ').title()}:**")
                    st.write(value)
    
    if patient_data.get("Dialogue"):
        st.subheader("🗣️ Conversation History")
        with st.expander("View Full Conversation"):
            for entry in patient_data["Dialogue"]:
                st.write(f"**{entry['speaker']}:** {entry['text']}")

def display_drug_quick_facts(drug_data):
    """Display drug quick facts in sidebar"""
    st.subheader("📊 Quick Facts")
    
    # Extract key information
    if 'openfda' in drug_data:
        openfda = drug_data['openfda']
        if 'brand_name' in openfda:
            st.write(f"**Brand Name:** {openfda['brand_name'][0] if openfda['brand_name'] else 'N/A'}")
        if 'generic_name' in openfda:
            st.write(f"**Generic Name:** {openfda['generic_name'][0] if openfda['generic_name'] else 'N/A'}")
        if 'manufacturer_name' in openfda:
            st.write(f"**Manufacturer:** {openfda['manufacturer_name'][0] if openfda['manufacturer_name'] else 'N/A'}")

def display_main_drug_info(drug_data):
    """Display main drug information"""
    priority_fields = ['indications_and_usage', 'dosage_and_administration', 'warnings', 'contraindications']
    
    for field in priority_fields:
        if field in drug_data:
            st.subheader(field.replace('_', ' ').title())
            if isinstance(drug_data[field], list):
                for item in drug_data[field]:
                    st.write(item)
            else:
                st.write(drug_data[field])
            st.divider()

def display_drug_details_enhanced(drug_name: str):
    """Enhanced drug details with better formatting and additional info using cached data"""
    st.title(f"💊 Drug Information: {drug_name}")
    
    # Check if drug data is in cache
    drug_data = st.session_state.get("drug_cache", {}).get(drug_name)
    
    if not drug_data:
        # Fallback to fetching if not in cache
        with st.spinner("Fetching drug information..."):
            drug_data = get_drug_purpose_with_retry(drug_name)
    
    if "error" in drug_data:
        st.error(drug_data["error"])
        st.info("💡 Try searching for alternative names or generic equivalents")
        
        # Suggest common alternatives
        st.subheader("🔍 Search Suggestions")
        st.write("Consider searching for:")
        st.write("• Generic name instead of brand name")
        st.write("• Brand name instead of generic name")
        st.write("• Alternative spellings")
        return
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_main_drug_info(drug_data)
    
    with col2:
        display_drug_quick_facts(drug_data)
    
    # Add drug safety warnings
    st.error("⚠️ **Important:** Always consult with a healthcare provider before starting or stopping medications")
    st.warning("📋 This information is for educational purposes only and should not replace professional medical advice")

def collect_user_feedback():
    """Collect user feedback on extracted information"""
    with st.expander("📝 Feedback & Corrections"):
        st.write("Help us improve by providing feedback on the extracted information:")
        
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Information Accuracy", "Missing Information", "Technical Issue"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Information is Accurate"):
                st.success("Thank you for confirming the accuracy!")
                logger.info("User confirmed information accuracy")
        
        with col2:
            if st.button("❌ Information Needs Correction"):
                st.session_state["feedback_mode"] = True
        
        if st.session_state.get("feedback_mode"):
            feedback_text = st.text_area("Please describe the corrections needed or provide additional details:")
            if st.button("Submit Feedback"):
                if feedback_text:
                    # Log the feedback
                    timestamp = datetime.now().isoformat()
                    logger.info(f"User feedback [{timestamp}] - {feedback_type}: {feedback_text}")
                    st.success("Thank you for your feedback! It helps us improve the system.")
                    st.session_state["feedback_mode"] = False
                else:
                    st.warning("Please provide feedback details before submitting.")

def display_patient_info_enhanced(patient_data):
    """Enhanced patient info display with tabs"""
    st.title("🏥 Patient Information Summary")
    
    # Search functionality
    st.subheader("🔍 Search Patient History")
    search_term = st.text_input("Search for specific terms in patient history:")
    if search_term:
        results = search_patient_history(patient_data, search_term)
        if results:
            st.subheader("Search Results")
            for result in results:
                st.write(f"• {result}")
        else:
            st.info("No results found for the search term.")
    
    # Export functionality
    st.subheader("📤 Export Patient Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as JSON"):
            json_data = export_patient_data(patient_data, "json")
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export as CSV"):
            csv_data = export_patient_data(patient_data, "csv")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Export as Text"):
            text_data = export_patient_data(patient_data, "text")
            st.download_button(
                label="Download Text",
                data=text_data,
                file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    st.divider()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Patient Info", "💊 Medications", "🧪 Tests & Results", "📋 History"])
    
    with tab1:
        display_basic_info(patient_data)
    
    with tab2:
        display_medications_tab(patient_data)
    
    with tab3:
        display_tests_tab(patient_data)
    
    with tab4:
        display_history_tab(patient_data)
    
    # Feedback section
    collect_user_feedback()

def main():
    st.set_page_config(
        page_title="Enhanced Patient Information System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with app info and settings
    with st.sidebar:
        st.title("🏥 Patient System")
        st.markdown("---")
        
        st.subheader("📊 System Status")
        st.success("✅ System Online")
        
        if config.debug_mode:
            st.info("🔧 Debug Mode: ON")
        
        st.subheader("⚙️ Settings")
        st.write(f"API Timeout: {config.api_timeout}s")
        st.write(f"Max Retries: {config.max_retries}")
        
        st.markdown("---")
        st.subheader("📖 Help")
        st.markdown("""
        **Navigation:**
        - Use tabs to explore different sections
        - Click 'Details' buttons for drug information
        - Use search to find specific information
        - Export data in multiple formats
        
        **Features:**
        - Drug interaction checking
        - Enhanced error handling
        - Data validation
        - User feedback system
        """)
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
    if "feedback_mode" not in st.session_state:
        st.session_state["feedback_mode"] = False
    
    # Process the predefined conversation
    with st.spinner("Processing patient conversation and fetching drug information..."):
        patient_data = extract_patient_info(record_list)
    
    # Page navigation
    if st.session_state["page"] == "main":
        if patient_data:
            display_patient_info_enhanced(patient_data)
        else:
            st.error("❌ Failed to extract patient information. Please check the logs and try again.")
            st.info("💡 Make sure your API keys are properly configured in the .env file")
    
    elif st.session_state["page"] == "drug_details" and "current_drug" in st.session_state:
        display_drug_details_enhanced(st.session_state["current_drug"])
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅️ Back to Patient Info"):
                st.session_state["page"] = "main"
        
        with col2:
            st.write("")  # Spacer

if __name__ == "__main__":
    main()