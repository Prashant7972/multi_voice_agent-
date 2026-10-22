import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge_base import KnowledgeBaseManager, extract_text_from_docx, chunk_text

def test_chunking():
    sample_text = "word " * 600
    chunks = chunk_text(sample_text, chunk_size_words=200, overlap_words=40)
    assert len(chunks) >= 3
    assert chunks[0]["chunk_id"] == "chunk-1"
    assert chunks[0]["word_count"] == 200

def test_knowledge_base_crud_and_isolation():
    kbm = KnowledgeBaseManager.get_instance()
    
    org_a = "test-security-org"
    org_b = "test-marketing-org"

    # Ingest text source in Org A
    src_a = kbm.add_text_source(
        org_id=org_a,
        title="Firewall Security Policy",
        content="All incoming SSH traffic on port 22 must be strictly routed through the enterprise VPN bastion gateway. Unauthenticated requests are immediately dropped.",
        category="Security",
        added_by="admin@sec.org"
    )
    assert src_a["id"].startswith("kb-")
    assert src_a["org_id"] == org_a

    # Ingest text source in Org B
    src_b = kbm.add_text_source(
        org_id=org_b,
        title="Social Media Brand Guidelines",
        content="Our primary brand colors are deep indigo and neon cyan. All public social media posts require two internal marketing review approvals.",
        category="Branding",
        added_by="admin@market.org"
    )
    assert src_b["org_id"] == org_b

    # Verify listing is scoped
    sources_a = kbm.list_sources(org_a)
    assert any(s["id"] == src_a["id"] for s in sources_a)
    assert not any(s["id"] == src_b["id"] for s in sources_a)

    sources_b = kbm.list_sources(org_b)
    assert any(s["id"] == src_b["id"] for s in sources_b)
    assert not any(s["id"] == src_a["id"] for s in sources_b)

    # Search in Org A for "SSH bastion gateway"
    results_a = kbm.search(org_id=org_a, query="SSH bastion gateway", top_k=3)
    assert len(results_a) > 0
    assert results_a[0]["source_id"] == src_a["id"]
    assert "bastion" in results_a[0]["text"].lower()

    # CRITICAL ISOLATION: Search in Org B for "SSH bastion gateway" MUST return 0 results
    results_b_isolation = kbm.search(org_id=org_b, query="SSH bastion gateway", top_k=3)
    assert not any(r["source_id"] == src_a["id"] for r in results_b_isolation)

    # Search in Org B for "neon cyan"
    results_b = kbm.search(org_id=org_b, query="neon cyan brand color", top_k=3)
    assert len(results_b) > 0
    assert results_b[0]["source_id"] == src_b["id"]

    # Delete source in Org A
    deleted = kbm.delete_source(org_a, src_a["id"])
    assert deleted is True
    assert kbm.get_source(org_a, src_a["id"]) is None

    # Cleanup Org B
    kbm.delete_source(org_b, src_b["id"])

def test_document_ingestion():
    kbm = KnowledgeBaseManager.get_instance()
    org_id = "test-doc-org"

    # Ingest text file
    txt_bytes = b"Server deployment protocol: Run docker compose up with standard production configs."
    src = kbm.add_document_source(
        org_id=org_id,
        filename="deployment_guide.txt",
        file_bytes=txt_bytes,
        category="DevOps",
        added_by="lead@devops.org"
    )
    assert src["source_type"] == "document"
    assert "docker compose" in src["content"]

    # RAG Context generation
    context, citations = kbm.build_rag_context(org_id, query="How do I deploy server?")
    assert "deployment" in context.lower()
    assert len(citations) > 0

    # Cleanup
    kbm.delete_source(org_id, src["id"])
    print("✅ Knowledge Base unit tests passed successfully!")

def test_agent_specific_knowledge_isolation():
    kbm = KnowledgeBaseManager.get_instance()
    org_id = "test-agent-isolation-org"
    agent_a = "sales-agent-007"
    agent_b = "finance-agent-999"

    # Add confidential source for Agent A
    src_a = kbm.add_text_source(
        org_id=org_id,
        title="Agent A Confidential Sales Playbook",
        content="Secret discount codes: DISCOUNT_VIP_50 for approved clients only. Never share with finance department.",
        category="Confidential Sales",
        added_by="admin@corp.org",
        agent_id=agent_a
    )
    assert src_a["agent_id"] == agent_a

    # Add confidential source for Agent B
    src_b = kbm.add_text_source(
        org_id=org_id,
        title="Agent B Confidential Audit Protocol",
        content="Internal escrow ledger balance is audited every Tuesday using MasterKeyAlpha999.",
        category="Confidential Finance",
        added_by="admin@corp.org",
        agent_id=agent_b
    )
    assert src_b["agent_id"] == agent_b

    # Verify listing with agent_id filter
    sources_a = kbm.list_sources(org_id, agent_id=agent_a)
    assert any(s["id"] == src_a["id"] for s in sources_a)
    assert not any(s["id"] == src_b["id"] for s in sources_a)

    sources_b = kbm.list_sources(org_id, agent_id=agent_b)
    assert any(s["id"] == src_b["id"] for s in sources_b)
    assert not any(s["id"] == src_a["id"] for s in sources_b)

    # Search query for Agent A's secret discount
    # When querying for Agent A, should find it
    results_for_a = kbm.search(org_id=org_id, query="DISCOUNT_VIP_50 secret discount codes", top_k=3, agent_id=agent_a)
    assert len(results_for_a) > 0
    assert results_for_a[0]["source_id"] == src_a["id"]
    assert results_for_a[0]["agent_id"] == agent_a

    # STRICT ISOLATION: When querying the exact same text for Agent B, it MUST NEVER find Agent A's confidential code
    results_for_b_leak_check = kbm.search(org_id=org_id, query="DISCOUNT_VIP_50 secret discount codes", top_k=3, agent_id=agent_b)
    assert len(results_for_b_leak_check) == 0, "Agent B must NOT access Agent A's confidential knowledge base!"

    # Context generation check
    ctx_a, cite_a = kbm.build_rag_context(org_id, query="What is the discount code?", agent_id=agent_a)
    assert "DISCOUNT_VIP_50" in ctx_a
    assert any(c["source_id"] == src_a["id"] for c in cite_a)

    ctx_b, cite_b = kbm.build_rag_context(org_id, query="What is the discount code?", agent_id=agent_b)
    assert "DISCOUNT_VIP_50" not in ctx_b
    assert len(cite_b) == 0

    # Cleanup
    kbm.delete_source(org_id, src_a["id"])
    kbm.delete_source(org_id, src_b["id"])
    print("✅ Agent-specific knowledge base confidentiality isolation verified successfully!")

def test_source_retrieval_distinction():
    kbm = KnowledgeBaseManager.get_instance()
    org_id = "test-retrieval-distinction-org"

    src_cloud = kbm.add_text_source(
        org_id=org_id,
        title="Enterprise Cloud Architecture",
        content="Apex AI systems deploy with isolated tenant containers and zero-latency Silero voice detection buffers.",
        category="Architecture",
        agent_id="tech-assistant-1"
    )

    src_sla = kbm.add_text_source(
        org_id=org_id,
        title="Apex AI Support Operations",
        content="Customer Support SLAs: Critical telephony tickets have a 15-minute response SLA.",
        category="Operations",
        agent_id="support-agent-2"
    )

    # 1. Matching query on cloud doc
    res_match = kbm.test_source_retrieval(org_id, src_cloud["id"], agent_id="tech-assistant-1", test_query="isolated tenant containers")
    assert res_match["is_accessible"] is True
    assert res_match["has_matching_chunks"] is True
    assert len(res_match["matched_chunks"]) > 0

    # 2. Unmatched query on cloud doc (asking about SLAs)
    res_unmatched = kbm.test_source_retrieval(org_id, src_cloud["id"], agent_id="tech-assistant-1", test_query="customer support SLA")
    assert res_unmatched["is_accessible"] is True
    assert res_unmatched["has_matching_chunks"] is False
    assert len(res_unmatched["matched_chunks"]) == 0
    assert "contains no content matching" in res_unmatched["message"]
    # Should hint that SLAs are in Support Operations
    assert "Support Operations" in res_unmatched["hint"]

    # 3. Confidential isolation: tech assistant accessing support doc
    res_isolated = kbm.test_source_retrieval(org_id, src_sla["id"], agent_id="tech-assistant-1", test_query="customer support SLA")
    assert res_isolated["is_accessible"] is False
    assert res_isolated["present_in_agent"] is False
    assert "Isolation" in res_isolated["message"]

    # Cleanup
    kbm.delete_source(org_id, src_cloud["id"])
    kbm.delete_source(org_id, src_sla["id"])

if __name__ == "__main__":
    test_chunking()
    test_knowledge_base_crud_and_isolation()
    test_document_ingestion()
    test_agent_specific_knowledge_isolation()
    test_source_retrieval_distinction()
