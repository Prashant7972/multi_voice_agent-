import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import AgentManager

def test_agent_manager_crud():
    print("🧪 Testing AgentManager...")
    mgr = AgentManager.get_instance()
    
    # 1. List agents
    agents = mgr.list_agents()
    print(f"   Found {len(agents)} agents.")
    assert len(agents) >= 1, "Should have at least 1 default agent"
    
    # Check General Voice Assistant is preserved
    names = [ag["name"] for ag in agents]
    print(f"   Agent names: {names}")
    assert "General Voice Assistant" in names, "General Voice Assistant must be preserved"
    
    # 2. Create new agent
    new_agent = mgr.create_agent({
        "name": "Finance Guru",
        "role": "Financial Advisor",
        "goals": "Advise on budget and stocks.",
        "capabilities": ["Web Search", "Report Generation"]
    })
    print(f"   Created agent: {new_agent['id']} - {new_agent['name']}")
    assert new_agent["id"].startswith("finance-guru")
    assert "Financial Advisor" in new_agent["system_prompt"]
    
    # 3. Activate agent
    activated = mgr.set_active_agent(new_agent["id"])
    assert activated["is_active"] is True
    assert mgr.get_active_agent()["id"] == new_agent["id"]
    print("   ✅ Agent activation verified.")
    
    # 4. Update agent
    updated = mgr.update_agent(new_agent["id"], {"goals": "Optimize investment portfolios."})
    assert "Optimize investment portfolios." in updated["goals"]
    print("   ✅ Agent update verified.")
    
    # 5. Delete agent
    deleted = mgr.delete_agent(new_agent["id"])
    assert deleted is True
    assert mgr.get_agent(new_agent["id"]) is None
    print("   ✅ Agent deletion verified.")
    
    # Restore general voice assistant as active
    mgr.set_active_agent("general-voice-assistant")
    print("🎉 All AgentManager tests passed successfully!")

if __name__ == "__main__":
    test_agent_manager_crud()
