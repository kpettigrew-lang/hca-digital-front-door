"""Automated Golden Dataset Test Suite & Clinical Benchmark Evaluation Harness."""

import json
import pytest
from agent import DigitalFrontDoorAgent, screen_clinical_red_flags, find_care_facilities, get_facility_capabilities

def test_tool_pydantic_validation():
    """Validates that tools handle invalid schema inputs with recovery guidance."""
    # Invalid zip code schema format
    result = find_care_facilities(zip_code="invalid-zip", care_tier="Urgent Care")
    assert "error" in result
    assert "guidance_for_agent" in result

def test_red_flag_critical_detection():
    """Validates that immediate life-threats return CRITICAL_EMERGENCY status."""
    res = screen_clinical_red_flags("Patient has sudden chest pain and slurred speech")
    assert res["status"] == "CRITICAL_EMERGENCY"
    assert res["is_emergency"] is True

def test_golden_dataset_evaluation_harness():
    """Executes the standardized golden dataset benchmark against the care navigation agent."""
    with open("golden_dataset.json", "r") as f:
        cases = json.load(f)

    agent = DigitalFrontDoorAgent()
    passed_cases = 0

    for case in cases:
        agent.reset_session()
        response = agent.run_turn(case["prompt"])
        
        # Verify required clinical keywords appear in response
        matches = any(keyword.lower() in response.lower() for keyword in case["must_contain"])
        assert matches, f"Case {case['case_id']} failed. Response did not contain any of {case['must_contain']}"

        # Verify Human-In-The-Loop escalation was properly flagged for emergency cases
        if case["requires_hitl"]:
            assert agent.human_in_the_loop_triggered is True, f"Case {case['case_id']} required HITL escalation trigger."
            
        passed_cases += 1

    assert passed_cases == len(cases)
