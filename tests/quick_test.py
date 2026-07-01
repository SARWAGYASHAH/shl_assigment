"""Quick test of the running server endpoints."""
import requests
import json

BASE = "http://localhost:8000"

# Test 1: Health
print("=" * 50)
print("TEST 1: Health Check")
r = requests.get(f"{BASE}/health")
print(f"Status: {r.status_code}, Body: {r.json()}")

# Test 2: Vague query (should NOT recommend)
print("\n" + "=" * 50)
print("TEST 2: Vague Query")
r = requests.post(f"{BASE}/chat", json={
    "messages": [{"role": "user", "content": "I need an assessment"}]
}, timeout=30)
d = r.json()
print(f"Reply: {d['reply'][:200]}")
print(f"Recommendations: {len(d['recommendations'])} (expected 0)")

# Test 3: Specific query (should recommend)
print("\n" + "=" * 50)
print("TEST 3: Java Developer Query")
r = requests.post(f"{BASE}/chat", json={
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"},
        {"role": "assistant", "content": "What seniority level are you looking for?"},
        {"role": "user", "content": "Mid-level, around 4 years experience"}
    ]
}, timeout=30)
d = r.json()
print(f"Reply: {d['reply'][:200]}")
print(f"Recommendations: {len(d['recommendations'])}")
for rec in d["recommendations"]:
    print(f"  - {rec['name']} [{rec['test_type']}] {rec['url']}")

# Test 4: Off-topic (should refuse)
print("\n" + "=" * 50)
print("TEST 4: Off-Topic Refusal")
r = requests.post(f"{BASE}/chat", json={
    "messages": [{"role": "user", "content": "What is the weather today?"}]
}, timeout=30)
d = r.json()
print(f"Reply: {d['reply'][:200]}")
print(f"Recommendations: {len(d['recommendations'])} (expected 0)")

print("\n" + "=" * 50)
print("ALL TESTS DONE")
