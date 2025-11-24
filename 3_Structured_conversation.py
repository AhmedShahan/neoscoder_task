from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr
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
    Lifestyle_Details: Optional[Dict[str, str]] = Field(default=None, description="Details of the patient's lifestyle, including occupation, diet, exercise, sleep, and habits")
    Current_Medical_Tests_Ordered: Optional[str] = Field(default=None, description="Medical tests ordered during the visit")
    Previous_Medical_Test_History: Optional[str] = Field(default=None, description="Previous medical tests conducted")
    Follow_Up_Actions: Optional[List[str]] = Field(default=None, description="List of follow-up actions or recommendations")
    # Doctor_Name: Optional[str] = Field(default=None, description="Name of the consulting doctor")
    Emotional_State: Optional[str] = Field(default=None, description="Emotional state of the patient during the visit")
    Dialogue: Optional[List[Dict[str, str]]] = Field(default=None, description="List of dialogue lines with speaker and text")



prescription_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Patient Information Extractor. Respond in valid JSON only."),
        ("human", "{conversation}\n\n{format_instructions}")
    ])


from demo_conversation import record_list

conversation=record_list
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
parser = StrOutputParser()
patient_parser = PydanticOutputParser(pydantic_object=PatientInformation)

chain=prescription_prompt | model | parser

result=chain.invoke({
    "conversation": conversation,
    "format_instructions": patient_parser.get_format_instructions()
})

print(result)