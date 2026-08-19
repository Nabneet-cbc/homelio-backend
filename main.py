import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HomeLioCare AI Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class TranscriptionRequest(BaseModel):
    text: str

class PatientInquiryExtraction(BaseModel):
    clientName: Optional[str] = Field(default=None, description="Full name of the patient")
    preferredName: Optional[str] = Field(default=None, description="Preferred name or nickname")
    dob: Optional[str] = Field(default=None, description="Date of birth (YYYY-MM-DD format if possible)")
    ssn: Optional[str] = Field(default=None, description="SSN or MBI number")
    gender: Optional[str] = Field(default=None, description="Gender")
    phone: Optional[str] = Field(default=None, description="Patient's phone number")
    email: Optional[str] = Field(default=None, description="Patient's email address")
    address: Optional[str] = Field(default=None, description="Street address")
    city: Optional[str] = Field(default=None, description="City")
    state: Optional[str] = Field(default=None, description="State (2-letter abbreviation)")
    zip: Optional[str] = Field(default=None, description="Zip or postal code")
    maritalStatus: Optional[str] = Field(default=None, description="Marital status")
    livingSituation: Optional[str] = Field(default=None, description="Living situation")
    primaryCaregiverAtHome: Optional[str] = Field(default=None, description="Name and relationship of primary caregiver at home (if any)")

    contactName: Optional[str] = Field(default=None, description="Full name of the primary contact person")
    contactRelationship: Optional[str] = Field(default=None, description="Relationship to the patient")
    contactPhone: Optional[str] = Field(default=None, description="Phone number of the primary contact")
    contactEmail: Optional[str] = Field(default=None, description="Email of the primary contact")
    isPOA: Optional[bool] = Field(default=None, description="True if this contact is the Power of Attorney (POA)")
    isEmergencyContact: Optional[bool] = Field(default=None, description="True if this contact is the emergency contact")

    natureOfInquiry: Optional[str] = Field(default=None, description="Nature of the inquiry")
    servicesInterestedIn: Optional[List[str]] = Field(default=None, description="List of services interested in (e.g. Skilled Nursing, Physical Therapy, Home Health Aide, Hospice Care)")
    startOfCareDate: Optional[str] = Field(default=None, description="Requested start-of-care date")
    caregiverPreferences: Optional[str] = Field(default=None, description="Caregiver preferences (e.g., Female, Spanish speaking)")
    accessInstructionsAndPets: Optional[str] = Field(default=None, description="Home access instructions and pet information")
    notes: Optional[str] = Field(default=None, description="Additional notes or comments about the inquiry")
    priority: Optional[str] = Field(default=None, description="Priority of the inquiry")

    primaryInsurance: Optional[str] = Field(default=None, description="Name of the primary insurance provider")
    memberId: Optional[str] = Field(default=None, description="Member ID for the primary insurance")

    # Extra fields missed initially
    sourceType: Optional[str] = Field(default=None, description="Source Type")
    communicationChannel: Optional[str] = Field(default=None, description="Communication Channel")
    howHeard: Optional[str] = Field(default=None, description="How did you hear about us?")
    campaignSource: Optional[str] = Field(default=None, description="Campaign / Source")
    additionalSourceDetails: Optional[str] = Field(default=None, description="Additional Source Details")
    
    race: Optional[str] = Field(default=None, description="Race / Ethnicity")
    preferredLanguage: Optional[str] = Field(default=None, description="Preferred Language")
    interpreterNeeded: Optional[bool] = Field(default=None, description="Is an interpreter needed?")
    
    preferredVisitTime: Optional[str] = Field(default=None, description="Preferred Visit Time Windows")
    
    secondaryInsurance: Optional[str] = Field(default=None, description="Secondary Insurance provider")
    secondaryMemberId: Optional[str] = Field(default=None, description="Secondary Insurance Member ID")
    
    internalNotes: Optional[str] = Field(default=None, description="Internal Intake Notes")
    followUpRequired: Optional[Literal["Yes", "No"]] = Field(default=None, description="Is follow-up required? (Yes / No)")
    followUpDate: Optional[str] = Field(default=None, description="Follow-up Date (YYYY-MM-DD)")
    assignedTo: Optional[str] = Field(default=None, description="Assigned To coordinator name")
    
    currentOutcome: Optional[str] = Field(default=None, description="Current Outcome")
    nextAction: Optional[str] = Field(default=None, description="Next Action")
    outcomeComments: Optional[str] = Field(default=None, description="Outcome Comments")

    # The absolute final missing fields
    intakeCoordinator: Optional[str] = Field(default=None, description="Intake Coordinator")
    inquiryStatus: Optional[str] = Field(default=None, description="Inquiry Status")
    patientAddressUnknown: Optional[bool] = Field(default=None, description="Patient address unknown")
    contactMethod: Optional[str] = Field(default=None, description="Preferred Contact Method")
    bestTimeToContact: Optional[str] = Field(default=None, description="Best Time to Contact")
    insuranceNotAvailable: Optional[bool] = Field(default=None, description="Insurance information not available")

def perform_extraction(text: str):
    schema = PatientInquiryExtraction.schema_json()
    system_prompt = f"""You are an expert AI assistant for a home healthcare platform.
Your task is to extract patient and inquiry data from the provided speech transcription and return ONLY a flat JSON object matching the exact keys provided in the schema below.

CRITICAL INSTRUCTIONS:
1. ONLY extract information that is explicitly stated or strongly implied in the text.
2. DO NOT GUESS OR HALLUCINATE ANY DATA. If a field is not mentioned, you MUST leave it empty (null).
3. If you are not confident about a piece of information, LEAVE IT EMPTY.
4. Format dates as YYYY-MM-DD if possible, otherwise leave them as extracted.
5. Do NOT nest the JSON inside a "patient" or "data" object. Return the flat keys directly.

Schema to match:
{schema}
"""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcription to extract from:\\n{text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    extracted_json_str = completion.choices[0].message.content
    extracted_data = json.loads(extracted_json_str)
    
    if "patient" in extracted_data and isinstance(extracted_data["patient"], dict):
        extracted_data = extracted_data["patient"]
    elif "PatientInquiryExtraction" in extracted_data and isinstance(extracted_data["PatientInquiryExtraction"], dict):
        extracted_data = extracted_data["PatientInquiryExtraction"]

    validated_data = PatientInquiryExtraction(**extracted_data)
    return validated_data.dict()


@app.post("/extract-patient")
async def extract_patient_data(request: TranscriptionRequest):
    try:
        data = perform_extraction(request.text)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-and-extract")
async def transcribe_and_extract(audio: UploadFile = File(...)):
    try:
        # 1. Transcribe the audio using Whisper
        file_bytes = await audio.read()
        transcription = client.audio.transcriptions.create(
            file=(audio.filename, file_bytes),
            model="whisper-large-v3",
            response_format="json"
        )
        text = transcription.text

        # 2. Extract data from the text
        extracted_data = perform_extraction(text)
        
        # Return both the text (so the frontend can display it) and the extracted data
        return {
            "transcript": text,
            "extractedData": extracted_data
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

