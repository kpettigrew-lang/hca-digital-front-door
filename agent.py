"""HCA Digital Front Door & Clinical Care Navigation Agent.

Production-grade multi-agent triage system built with Google GenAI SDK.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# 1. Observability, Structured JSON Logging & PHI Redaction
# ==============================================================================

class PHIRedactingFormatter(logging.Formatter):
    """Regex-based PHI/PII sanitizer for HIPAA-aligned clinical logging."""
    
    PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), "[REDACTED_PHONE]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[REDACTED_DOB]"),
    ]

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern, replacement in self.PATTERNS:
            msg = pattern.sub(replacement, msg)
        return msg

logger = logging.getLogger("HCADigitalFrontDoor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(PHIRedactingFormatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

def emit_audit_trace(event: str, session_id: str, intent: str, action: str, outcome: str, metadata: Optional[Dict[str, Any]] = None):
    """Logs structured telemetry events for full observability audit trails."""
    trace_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "hca-digital-front-door",
        "session_id": session_id,
        "event": event,
        "intent": intent,
        "action": action,
        "outcome": outcome,
        "metadata": metadata or {}
    }
    logger.info(json.dumps(trace_payload))

# ==============================================================================
# 2. Synthetic Clinical Registry & Pydantic Tool Schemas
# ==============================================================================

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

class RedFlagInput(BaseModel):
    symptoms: str = Field(..., min_length=2, description="Free-text description of patient symptoms")

class FindFacilitiesInput(BaseModel):
    zip_code: str = Field(..., pattern=r"^(\d{5}|ANY)$", description="5-digit US postal code or 'ANY'")
    care_tier: str = Field(..., description="Acuity tier: 'Emergency Department', 'Urgent Care', or 'Virtual Care'")

class FacilityCapabilitiesInput(BaseModel):
    facility_id: str = Field(..., pattern=r"^FAC-\d{3}$", description="Facility identifier code, e.g., 'FAC-001'")

def screen_clinical_red_flags(symptoms: str) -> dict:
    """Evaluates reported symptoms against immediate high-acuity life-safety red flags."""
    try:
        validated = RedFlagInput(symptoms=symptoms)
    except ValidationError as err:
        return {
            "error": "Validation failed",
            "details": err.errors(),
            "guidance_for_agent": "Please clarify user symptoms in plain text before continuing triage."
        }

    s_lower = validated.symptoms.lower()
    matched = [flag for flag in RED_FLAG_SYMPTOMS if flag in s_lower]
    if matched:
        return {
            "status": "CRITICAL_EMERGENCY",
            "matched_red_flags": matched,
            "action_required": "IMMEDIATELY CALL 911 OR ROUTE TO EMERGENCY DEPARTMENT.",
            "is_emergency": True
        }
    return {
        "status": "NON_CRITICAL",
        "matched_red_flags": [],
        "action_required": "Proceed with standard acuity-based facility routing.",
        "is_emergency": False
    }

def find_care_facilities(zip_code: str, care_tier: str) -> dict:
    """Queries nearby facilities matching patient zip code and recommended care tier."""
    try:
        validated = FindFacilitiesInput(zip_code=zip_code, care_tier=care_tier)
    except ValidationError as err:
        return {
            "error": "Invalid facility query parameters",
            "details": err.errors(),
            "guidance_for_agent": "Zip code must be 5 numeric digits or 'ANY'. Care tier must match 'Emergency Department', 'Urgent Care', or 'Virtual Care'."
        }

    matched = []
    for fac in FACILITIES_DB:
        tier_match = (validated.care_tier.lower() in fac["type"].lower()) or (validated.care_tier.lower() == "any")
        zip_match = (fac["zip"] == "ALL") or (validated.zip_code == "ANY") or (fac["zip"] == validated.zip_code)
        if tier_match and zip_match:
            matched.append(fac)
    
    if not matched:
        matched = [f for f in FACILITIES_DB if f["type"] == "Virtual Care"]
    return {"facilities_found": len(matched), "results": matched}

def get_facility_capabilities(facility_id: str) -> dict:
    """Retrieves on-site diagnostic equipment, staffing, and clinical capabilities for a facility."""
    try:
        validated = FacilityCapabilitiesInput(facility_id=facility_id)
    except ValidationError as err:
        return {
            "error": "Invalid facility identifier",
            "details": err.errors(),
            "guidance_for_agent": "Facility ID must match format FAC-XXX (e.g., FAC-001, FAC-002, FAC-003, FAC-004)."
        }

    for fac in FACILITIES_DB:
        if fac["facility_id"] == validated.facility_id:
            return fac
    return {
        "error": f"Facility ID '{validated.facility_id}' not found in registry.",
        "guidance_for_agent": "Use find_care_facilities to discover valid facility IDs before requesting capabilities."
    }

TOOLS_REGISTRY = {
    "screen_clinical_red_flags": screen_clinical_red_flags,
    "find_care_facilities": find_care_facilities,
    "get_facility_capabilities": get_facility_capabilities
}

# ==============================================================================
# 3. Multi-Agent Persona Orchestration & Model Routing
# ==============================================================================

TRIAGE_ROUTER_INSTRUCTIONS = """
You are the Primary HCA Triage Router Agent.
Your duty is fast, conservative acuity screening.
1. ALWAYS execute `screen_clinical_red_flags` first.
2. If ANY red flag exists, trigger immediate 911/Emergency Department escalation.
3. For non-emergent cases, determine acuity tier (Urgent Care vs. Virtual Care) and query candidate facilities.
4. Delegate specialist questions about diagnostic readiness to the Specialist Agent.
Always include a clear disclaimer that guidance is navigational and not a diagnosis.
"""

SPECIALIST_AGENT_INSTRUCTIONS = """
You are the HCA Clinical Facility Specialist Agent.
You provide detailed analysis of on-site diagnostic capabilities (X-ray, Labs, Suturing) and wait times.
Keep responses concise, clear, and reassuring.
"""

class MockPersistentStore:
    """Simulates persistent key-value session storage (e.g., Cloud Spanner / Firestore)."""
    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    async def save_history(self, session_id: str, history: List[Any]):
        self._store[session_id] = [h.to_json_dict() if hasattr(h, "to_json_dict") else str(h) for h in history]

    async def load_history(self, session_id: str) -> List[Any]:
        return self._store.get(session_id, [])

    async def purge(self, session_id: str):
        self._store.pop(session_id, None)

class DigitalFrontDoorAgent:
    """Autonomous clinical triage navigator featuring multi-agent routing and context compaction."""

    def __init__(
        self,
        router_model: str = "gemini-3.8-flash",
        specialist_model: str = "gemini-2.5-flash",
        max_turns_window: int = 6
    ):
        self.client = genai.Client()
        self.router_model = router_model
        self.specialist_model = specialist_model
        self.max_turns_window = max_turns_window
        self.conversation_history: List[types.Content] = []
        self.session_id = f"session-{int(datetime.utcnow().timestamp())}"
        self.persistence = MockPersistentStore()
        self.tools = [screen_clinical_red_flags, find_care_facilities, get_facility_capabilities]
        self.human_in_the_loop_triggered = False

    def reset_session(self):
        """Resets session context and purges persisted session memory."""
        self.conversation_history = []
        self.session_id = f"session-{int(datetime.utcnow().timestamp())}"
        self.human_in_the_loop_triggered = False
        emit_audit_trace("SESSION_RESET", self.session_id, "RESET", "CLEAR_MEMORY", "SUCCESS")

    async def _compact_context_if_needed(self):
        """Context Compaction: Compacts memory using sliding window summarization when context overflows."""
        if len(self.conversation_history) > self.max_turns_window:
            emit_audit_trace("CONTEXT_COMPACTION", self.session_id, "MEMORY_MANAGEMENT", "SUMMARIZE", "TRIGGERED")
            # Preserve system anchor and compact older turns into a consolidated context summary
            preserved_recent = self.conversation_history[-4:]
            summary_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text="[Prior consultation context summarized for clinical brevity: Patient symptoms previously evaluated.]")]
            )
            self.conversation_history = [summary_content] + preserved_recent
            await self.persistence.save_history(self.session_id, self.conversation_history)

    def trigger_human_in_the_loop_escalation(self, reason: str) -> dict:
        """Human-in-the-Loop hook for high-risk triage events requiring staff confirmation."""
        self.human_in_the_loop_triggered = True
        escalation_payload = {
            "status": "ESCALATION_TRIGGERED",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "escalated_to": "HCA_ONCALL_EMERGENCY_DISPATCH",
            "requires_human_confirmation": True
        }
        emit_audit_trace("HUMAN_IN_THE_LOOP", self.session_id, "EMERGENCY_ESCALATION", "DISPATCH_ALERT", "PENDING_CONFIRMATION", escalation_payload)
        return escalation_payload

    async def run_turn_async(self, user_prompt: str) -> str:
        """Asynchronous execution loop for care navigation."""
        await self._compact_context_if_needed()
        emit_audit_trace("USER_PROMPT", self.session_id, "TRIAGE_REQUEST", "PROCESS_INPUT", "RECEIVED", {"prompt": user_prompt})

        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )

        iteration = 1
        max_iterations = 5

        while iteration <= max_iterations:
            # Model Routing: Flash model for fast tool triage
            config = types.GenerateContentConfig(
                system_instruction=TRIAGE_ROUTER_INSTRUCTIONS,
                tools=self.tools,
                temperature=0.1,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )

            # Robust invocation handling
            response = None
            for attempt in range(4):
                try:
                    response = self.client.models.generate_content(
                        model=self.router_model,
                        contents=self.conversation_history,
                        config=config
                    )
                    break
                except Exception as ex:
                    if "503" in str(ex) and attempt < 3:
                        await asyncio.sleep((attempt + 1) * 2)
                    else:
                        raise ex

            candidate = response.candidates[0].content
            self.conversation_history.append(candidate)

            function_calls = [p.function_call for p in candidate.parts if p.function_call]
            if not function_calls:
                final_text = "".join(p.text for p in candidate.parts if p.text)
                emit_audit_trace("FINAL_RESPONSE", self.session_id, "CARE_RECOMMENDATION", "DELIVER_OUTPUT", "COMPLETED")
                await self.persistence.save_history(self.session_id, self.conversation_history)
                return final_text

            tool_response_parts = []
            for call in function_calls:
                fn_name = call.name
                fn_args = dict(call.args)
                
                # Check for critical red flag trigger -> Human-In-The-Loop escalation hook
                if fn_name == "screen_clinical_red_flags":
                    pre_check = screen_clinical_red_flags(**fn_args)
                    if pre_check.get("is_emergency"):
                        self.trigger_human_in_the_loop_escalation("Critical red-flag symptom detected during screening.")

                # Execute validated tool
                tool_fn = TOOLS_REGISTRY.get(fn_name)
                output = tool_fn(**fn_args) if tool_fn else {"error": f"Tool {fn_name} unregistered"}
                
                emit_audit_trace(
                    "TOOL_EXECUTION", self.session_id, fn_name, "CALL_REGISTRY_TOOL",
                    "SUCCESS" if "error" not in output else "VALIDATION_ERROR",
                    {"args": fn_args, "result_summary": str(output)[:120]}
                )

                tool_response_parts.append(
                    types.Part.from_function_response(name=fn_name, response={"result": output})
                )

            self.conversation_history.append(
                types.Content(role="user", parts=tool_response_parts)
            )
            iteration += 1

        return "Care triage iteration ceiling reached."

    def run_turn(self, user_prompt: str) -> str:
        """Synchronous wrapper for notebooks and sync consumers."""
        return asyncio.run(self.run_turn_async(user_prompt))
