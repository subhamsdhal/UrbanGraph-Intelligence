"""UrbanGraph Intelligence API Client Example.
Demonstrates authenticated interaction with simulation and optimization endpoints.
"""
import requests
import json

BASE_URL = "http://localhost:8000"
TOKEN = "your_bearer_token_here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 1. Healthcheck
resp = requests.get(f"{BASE_URL}/api/v1/health")
print("Health status:", resp.json())

# 2. Get Grid Topology
resp = requests.get(f"{BASE_URL}/api/v1/grid/topology", headers=headers)
print("Grid topology nodes:", len(resp.json().get("buses", [])))

# 3. Trigger DC Power Flow Simulation
sim_payload = {
    "network_id": "eg006",
    "load_scaling": 1.05,
    "slack_bus": 1
}
resp = requests.post(f"{BASE_URL}/api/v1/grid/simulate", json=sim_payload, headers=headers)
print("Power flow converged:", resp.json().get("converged"))
print("Branch flows:", resp.json().get("branch_flows"))
