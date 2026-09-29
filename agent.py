import json
import os
import time
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client()
MODEL_ID = "gemini-3.8-flash"

# Synthetic Clinical Registry
FACILITIES_DB = [
    {
        "facility_id": "FAC-001",
        "name": "Tristar Centennial Medical Center - Emergency Room",
        "type": "Emergency Department",
        "zip": "37203",
        "wait_time_mins": 14,
        "capabilities": ["Trauma Center", "Cardiac Cath Lab", "CT/MRI", "24/7 Resuscitation"],
        "open_now": True
    },
    {
        "facility_id": "FAC-002",
        "name": "CareNow Urgent Care - Midtown Nashville",
        "type": "Urgent Care",
        "zip": "37203",
        "wait_time_mins": 25,
        "capabilities": ["X-ray", "Laceration Suturing", "Point-of-Care Lab", "Flu/Strep Swab"],
        "open_now": True
    },
    {
        "facility_id": "FAC-003",
        "name": "CareNow Urgent Care - Hendersonville",
        "type": "Urgent Care",
        "zip": "37075",
        "wait_time_mins": 10,
        "capabilities": ["X-ray", "Minor Splinting", "Urinalysis"],
        "open_now": True
    },
    {
        "facility_id": "FAC-004",
        "name": "HCA Virtual Care On-Demand Clinic",
        "type": "Virtual Care",
        "zip": "ALL",
        "wait_time_mins": 5,
        "capabilities": ["Telehealth", "Prescription Refill", "Rashes/Minor Infection"],
        "open_now": True
    }
]

RED_FLAG_SYMPTOMS = [
    "chest pain", "pressure in chest", "shortness of breath", "difficulty breathing",
    "stroke", "facial droop", "slurred speech", "uncontrolled bleeding", "loss of consciousness",
    "severe sudden headache", "anaphylaxis", "coughing blood"
]

def screen_clinical_red_flags(symptoms: str) -> dict:
    """Evaluates reported symptoms against immediate high-acuity life-safety red flags.
    
    Args:
        symptoms: Free-text description of the patient's symptoms.
        
    Returns:
        dict: Triage risk level ('CRITICAL_EMERGENCY' or 'NON_CRITICAL') and routing advice.
    """
    symptoms_lower = symptoms.lower()
    detected = [flag for flag in RED_FLAG_SYMPTOMS if flag in symptoms_lower]
    if detected:
        return {
            "status": "CRITICAL_EMERGENCY",
            "matched_red_flags": detected,
            "action_required": "IMMEDIATELY CALL 911 OR GO TO THE NEAREST EMERGENCY ROOM.",
            "is_emergency": True
        }
    return {
        "status": "NON_CRITICAL",
        "matched_red_flags": [],
        "action_required": "Proceed with standard facility search based on acuity.",
        "is_emergency": False
    }

def find_care_facilities(zip_code: str, care_tier: str) -> dict:
    """Queries nearby facilities matching patient zip code and recommended care tier.
    
    Args:
        zip_code: 5-digit zip code string (e.g., '37203') or 'ANY'.
        care_tier: Level of care needed: 'Emergency Department', 'Urgent Care', or 'Virtual Care'.
        
    Returns:
        dict: A list of candidate facilities with current estimated wait times.
    """
    matched = []
    for fac in FACILITIES_DB:
        tier_match = (care_tier.lower() in fac["type"].lower()) or (care_tier.lower() == "any")
        zip_match = (fac["zip"] == "ALL") or (zip_code == "ANY") or (fac["zip"] == zip_code)
        if tier_match and zip_match:
            matched.append(fac)
    if not matched:
        matched = [f for f in FACILITIES_DB if f["type"] == "Virtual Care"]
    return {"facilities_found": len(matched), "results": matched}

def get_facility_capabilities(facility_id: str) -> dict:
    """Retrieves on-site diagnostic equipment, staffing, and clinical capabilities for a facility.
    
    Args:
        facility_id: Unique facility identifier (e.g., 'FAC-001', 'FAC-002').
        
    Returns:
        dict: Detailed capabilities and diagnostic equipment available.
    """
    for fac in FACILITIES_DB:
        if fac["facility_id"] == facility_id:
            return fac
    return {"error": f"Facility ID '{facility_id}' not found."}

TOOLS_REGISTRY = {
    "screen_clinical_red_flags": screen_clinical_red_flags,
    "find_care_facilities": find_care_facilities,
    "get_facility_capabilities": get_facility_capabilities
}

SYSTEM_INSTRUCTIONS = """
You are the HCA Digital Front Door & Care Navigation Agent.
Your primary role is to guide patients to the most appropriate care setting: Emergency Department (ED), Urgent Care, or Virtual Care.

CLINICAL BOUNDARIES & GUARDRAILS:
1. ALWAYS execute `screen_clinical_red_flags` FIRST before searching for standard care facilities.
2. If ANY red flag is detected, immediately instruct the user to call 911 or visit the nearest ER. DO NOT suggest an Urgent Care or Virtual Visit for emergency symptoms.
3. You are a navigational assistant, NOT a medical doctor. Always include a brief disclaimer that this is navigational guidance, not an official medical diagnosis.
4. For non-emergent conditions requiring on-site services, find Urgent Care clinics and check their capabilities.
5. For mild, non-physical conditions, suggest Virtual Care.
"""

class DigitalFrontDoorAgent:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.conversation_history = []
        self.tools = [screen_clinical_red_flags, find_care_facilities, get_facility_capabilities]
        
    def reset_session(self):
        self.conversation_history = []

    def run_turn(self, user_prompt: str) -> str:
        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )
        
        iteration = 1
        while iteration <= 5:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                tools=self.tools,
                temperature=0.1,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
            
            response = client.models.generate_content(
                model=self.model_id,
                contents=self.conversation_history,
                config=config
            )
            
            candidate_content = response.candidates[0].content
            self.conversation_history.append(candidate_content)
            
            function_calls = [part.function_call for part in candidate_content.parts if part.function_call]
            if not function_calls:
                return "".join(part.text for part in candidate_content.parts if part.text)
                
            tool_response_parts = []
            for call in function_calls:
                fn_name = call.name
                fn_args = dict(call.args)
                tool_output = TOOLS_REGISTRY.get(fn_name, lambda **x: {"error": "not found"})(**fn_args)
                
                # Observability trace logging
                print(f"[TRACE] Executed {fn_name} with {fn_args} -> Result: {tool_output}")
                
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": tool_output}
                    )
                )
                
            self.conversation_history.append(
                types.Content(role="user", parts=tool_response_parts)
            )
            iteration += 1
        return "Iteration limit reached."
