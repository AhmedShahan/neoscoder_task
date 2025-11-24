import streamlit as st
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from dotenv import load_dotenv

# Load environment
load_dotenv()

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

def extract_patient_info(conversation):
    """Extract patient information from conversation"""
    prescription_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Patient Information Extractor. Extract relevant patient information from the conversation and respond with valid JSON matching the provided format. Ensure the output is a valid JSON string and contains only the fields specified in the format instructions."),
        ("human", "{conversation}\n\n{format_instructions}")
    ])
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)  # Lower temperature for more consistent output
    patient_parser = PydanticOutputParser(pydantic_object=PatientInformation)
    
    chain = prescription_prompt | model | patient_parser  # Use PydanticOutputParser directly
    
    result = chain.invoke({
        "conversation": conversation,
        "format_instructions": patient_parser.get_format_instructions()
    })
    
    return result.dict()  # Return the parsed Pydantic model as a dictionary

def display_patient_info(patient_data):
    """Display patient information in a formatted way"""
    
    # Header
    st.title("🏥 Patient Information Summary")
    st.divider()
    
    # Basic Information
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
    
    # Current Health Issue
    if patient_data.get("Recent_Problem"):
        st.subheader("🩺 Current Health Issue")
        st.write(patient_data["Recent_Problem"])
        st.divider()
    
    # Medical History Section
    col1, col2 = st.columns(2)
    
    with col1:
        if patient_data.get("Previous_Medical_History"):
            st.subheader("📋 Medical History")
            st.write(patient_data["Previous_Medical_History"])
        
        if patient_data.get("Previous_Drug_History"):
            st.subheader("💊 Drug History")
            st.write(patient_data["Previous_Drug_History"])
    
    with col2:
        if patient_data.get("Allergies"):
            st.subheader("⚠️ Allergies")
            st.error(patient_data["Allergies"])
        
        if patient_data.get("Family_Medical_History"):
            st.subheader("👨‍👩‍👧‍👦 Family History")
            st.write(patient_data["Family_Medical_History"])
    
    st.divider()
    
    # Lifestyle Details
    if patient_data.get("Lifestyle_Details"):
        st.subheader("🏃‍♂️ Lifestyle Information")
        lifestyle = patient_data["Lifestyle_Details"]
        
        cols = st.columns(len(lifestyle))
        for i, (key, value) in enumerate(lifestyle.items()):
            with cols[i]:
                st.write(f"**{key.replace('_', ' ').title()}:**")
                st.write(value)
        
        st.divider()
    
    # Medical Tests
    col1, col2 = st.columns(2)
    
    with col1:
        if patient_data.get("Current_Medical_Tests_Ordered"):
            st.subheader("🧪 Tests Ordered")
            st.info(patient_data["Current_Medical_Tests_Ordered"])
    
    with col2:
        if patient_data.get("Previous_Medical_Test_History"):
            st.subheader("📊 Previous Tests")
            st.write(patient_data["Previous_Medical_Test_History"])
    
    # Follow-up Actions
    if patient_data.get("Follow_Up_Actions"):
        st.subheader("📝 Follow-up Actions")
        for action in patient_data["Follow_Up_Actions"]:
            st.write(f"• {action}")
        st.divider()
    
    # Emotional State
    if patient_data.get("Emotional_State"):
        st.subheader("😊 Emotional State")
        st.write(patient_data["Emotional_State"])

def main():
    st.set_page_config(
        page_title="Patient Information System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Sidebar for input
    st.sidebar.header("📝 Input Conversation")
    
    # Text area for conversation input
    conversation_input = st.sidebar.text_area(
        "Enter patient conversation:",
        height=400,
        placeholder="Paste the patient conversation here..."
    )
    
    # Process button
    if st.sidebar.button("Extract Patient Information", type="primary"):
        if conversation_input.strip():
            try:
                with st.spinner("Processing conversation..."):
                    patient_data = extract_patient_info(conversation_input)
                    st.session_state.patient_data = patient_data
                    st.success("Patient information extracted successfully!")
            except Exception as e:
                st.error(f"Error processing conversation: {str(e)}")
        else:
            st.warning("Please enter a conversation to process.")
    
    # Display patient information if available
    if 'patient_data' in st.session_state:
        display_patient_info(st.session_state.patient_data)
    else:
        st.info("👈 Enter a patient conversation in the sidebar to extract information.")

if __name__ == "__main__":
    main()