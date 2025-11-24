import streamlit as st
import os
import requests
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Debug: Print API key to verify
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

# Provided conversation list
from demo_conversation import recorded_list2
record_list = recorded_list2

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

# New Pydantic model for drug names extraction
class DrugNames(BaseModel):
    drug_names: List[str] = Field(description="List of clean drug names extracted from drug history, containing only the medication names without dosages, frequencies, or additional information")

def extract_drug_names(drug_history: str) -> List[str]:
    """Extract clean drug names from drug history text using LLM"""
    try:
        # Initialize the model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
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
        
        return result.drug_names
        
    except Exception as e:
        st.error(f"Error extracting drug names: {str(e)}")
        # Fallback to regex-based extraction
        return extract_drug_names_fallback(drug_history)

def extract_drug_names_fallback(drug_history: str) -> List[str]:
    """Fallback method to extract drug names using regex"""
    if not drug_history:
        return []
    
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

def get_drug_purpose(drug_name: str) -> Dict:
    """Gets the drug purpose and details from OpenFDA API."""
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"Error: {response.status_code}"}
        data = response.json()
        results = data.get('results', [])
        if not results:
            return {"error": "No results found for the drug."}
        return results[0]
    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}

def extract_patient_info(conversation):
    """Extract patient information from conversation"""
    try:
        # Initialize the model with the same settings as the working code
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
        
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
        
        return result.dict()
    except Exception as e:
        st.error(f"Error processing conversation: {str(e)}")
        return None

def display_drug_details(drug_name: str):
    """Display detailed drug information in a new page"""
    st.title(f"💊 Drug Information: {drug_name}")
    st.divider()
    
    drug_data = get_drug_purpose(drug_name)
    
    if "error" in drug_data:
        st.error(drug_data["error"])
        return
    
    for key, value in drug_data.items():
        if isinstance(value, dict):
            st.subheader(key.replace('_', ' ').title())
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    st.write(f"**{sub_key.replace('_', ' ').title()}:**")
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        st.write(f"  • {sub_sub_key.replace('_', ' ').title()}: {sub_sub_value}")
                else:
                    st.write(f"**{sub_key.replace('_', ' ').title()}:** {sub_value}")
        elif isinstance(value, list):
            st.subheader(key.replace('_', ' ').title())
            for item in value:
                st.write(f"• {item}")
        else:
            st.subheader(key.replace('_', ' ').title())
            st.write(value)
        st.divider()

def display_patient_info(patient_data):
    """Display patient information in a formatted way"""
    st.title("🏥 Patient Information Summary")
    st.divider()
    
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
    
    st.divider()
    
    if patient_data.get("Recent_Problem"):
        st.subheader("🩺 Current Health Issue")
        st.write(patient_data["Recent_Problem"])
        st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if patient_data.get("Previous_Medical_History"):
            st.subheader("📋 Medical History")
            st.write(patient_data["Previous_Medical_History"])
        
        if patient_data.get("Previous_Drug_History"):
            st.subheader("💊 Drug History")
            st.write(patient_data["Previous_Drug_History"])
            
            # Extract clean drug names using the new LLM-based method
            drug_names = extract_drug_names(patient_data["Previous_Drug_History"])
            
            # Display extracted drug names for debugging
            st.write("**Extracted Drug Names:**")
            for drug in drug_names:
                st.write(f"• {drug}")
            
            # Add Explain button for each drug
            for drug in drug_names:
                if st.button(f"Explain: {drug}", key=f"explain_{drug}"):
                    st.session_state["current_drug"] = drug
                    st.session_state["page"] = "drug_details"
    
    with col2:
        if patient_data.get("Allergies"):
            st.subheader("⚠️ Allergies")
            st.error(patient_data["Allergies"])
        
        if patient_data.get("Family_Medical_History"):
            st.subheader("👨‍👩‍👧‍👦 Family History")
            st.write(patient_data["Family_Medical_History"])
    
    st.divider()
    
    if patient_data.get("Lifestyle_Details"):
        st.subheader("🏃‍♂️ Lifestyle Information")
        lifestyle = patient_data["Lifestyle_Details"]
        
        cols = st.columns(len(lifestyle))
        for i, (key, value) in enumerate(lifestyle.items()):
            with cols[i]:
                st.write(f"**{key.replace('_', ' ').title()}:**")
                st.write(value)
        
        st.divider()
    
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
        st.divider()
    
    if patient_data.get("Emotional_State"):
        st.subheader("😊 Emotional State")
        st.write(patient_data["Emotional_State"])
    
    if patient_data.get("Dialogue"):
        st.subheader("🗣️ Conversation")
        for entry in patient_data["Dialogue"]:
            st.write(f"**{entry['speaker']}:** {entry['text']}")
        st.divider()

def main():
    st.set_page_config(
        page_title="Patient Information System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
    
    # Process the predefined conversation
    patient_data = extract_patient_info(record_list)
    
    # Page navigation
    if st.session_state["page"] == "main":
        if patient_data:
            display_patient_info(patient_data)
        else:
            st.info("No patient information available. Check API key and try again.")
    elif st.session_state["page"] == "drug_details" and "current_drug" in st.session_state:
        display_drug_details(st.session_state["current_drug"])
        if st.button("Back to Patient Info"):
            st.session_state["page"] = "main"

if __name__ == "__main__":
    main()