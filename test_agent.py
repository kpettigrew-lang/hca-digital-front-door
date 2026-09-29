from agent import DigitalFrontDoorAgent

def test_emergency_red_flag_routing():
    agent = DigitalFrontDoorAgent()
    agent.reset_session()
    result = agent.run_turn("Severe chest pressure and shortness of breath in 37203.")
    assert "911" in result or "Emergency" in result

def test_urgent_care_routing():
    agent = DigitalFrontDoorAgent()
    agent.reset_session()
    result = agent.run_turn("Twisted ankle playing soccer in 37203, hurts to bear weight.")
    assert "Urgent Care" in result or "CareNow" in result

def test_virtual_care_routing():
    agent = DigitalFrontDoorAgent()
    agent.reset_session()
    result = agent.run_turn("Mild skin rash on my arm for 2 days, no fever in 37075.")
    assert "Virtual" in result or "Urgent Care" in result
