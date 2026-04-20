# AC-DL-MATCH: Adaptive Context-Aware Distributed Learning Matching

> **Adaptive Context-Aware Task Offloading with Dynamic Infrastructure Elasticity in Multi-Domain SDN-Enabled Fog Computing Networks**

## 📋 Project Overview

**Type**: Research Paper & Simulation Project  
**Domain**: Fog Computing, Task Offloading, Infrastructure Elasticity  
**Implementation**: Python Benchmarking Suite (PyTorch, Numpy)  
**Team**: Pratham Aggarwal (22JE0718)

---

## 🎯 Problem Statement

Modern fog computing systems face **critical failures** that prevent real-world deployment at scale:

### 1. **Static/Rule-Based Task Offloading**
Current algorithms use fixed rules that fail when fog nodes become overloaded, slow down, or crash—leading to high latency and task failures during traffic spikes.

**Example**: During a traffic accident, hundreds of autonomous vehicles suddenly need real-time path computation. Static systems can't adapt, causing dangerous delays.

### 2. **No Infrastructure Adaptation**
Systems assume a fixed number of fog nodes. They **cannot automatically**:
- Add fog nodes during demand spikes
- Remove idle resources to save energy

**Consequence**: Manual provisioning takes hours/days while IoT devices suffer immediate failures.

### 3. **Single-Objective Optimization**
Current approaches consider only delay and cost, ignoring:
- Energy consumption (critical for battery-powered sensors)
- Reliability (uptime requirements vary: healthcare 99.999% vs. video streaming 95%)
- Application-specific priorities (heart monitors vs. weather apps)

### 4. **Lack of Scalability**
- **Centralized optimization (ILP)**: Cannot handle 10,000+ IoT devices
- **Deep Learning (DRL)**: Requires 100,000+ training iterations, excessive computational power, unsuitable for resource-constrained ESP32/Raspberry Pi devices

### 5. **Single-Domain Deadlock**
Tasks cannot migrate between fog provider domains (e.g., from Provider A to Provider B) when local capacity exhausted, causing **complete system failures**.

### 6. **Dimensional Collapse in Utilities**
Traditional multi-objective functions suffer from *dimensional collapse*, where raw metrics of wildly varying magnitudes (e.g., `100ms` latency vs `$0.01` cost) disproportionately skew the optimization gradients. Resolving this via global Z-scores violates decentralized zero-communication topologies.

---

## 💡 Proposed Solution: AC-DL-MATCH

**AC-DL-MATCH** (Adaptive Context-Aware Distributed Learning Matching) is a distributed framework integrating intelligent task offloading with dynamic infrastructure elasticity.

### Key Innovations

#### 1️⃣ **Context-Aware Multi-Objective Utility Function**

Adapts optimization weights based on **task type**:

| Task Type | Priority | Weight Focus |
|-----------|----------|--------------|
| Autonomous Vehicles | Delay-sensitive | High ω_D (delay weight) |
| Battery Sensors | Energy-critical | High ω_E (energy weight) |
| Healthcare Monitoring | Mission-critical | High ω_R (reliability weight) |

**Formula**:
```
U_ij^modified = ω_1^Ti × (1/D_ij) + ω_2^Ti × (1/E_ij) + ω_3^Ti × R_j - ω_4^Ti × C_ij

Where:
- D_ij = Total delay (propagation + transmission + execution)
- E_ij = Energy consumption (P_tran × (S_i / r_ij) + P_idle × Q_j)
- R_j = Fog node reliability score (Uptime / TotalTime)
- C_ij = Monetary cost (C_j × S_i)
- ω^Ti = Task-specific weights (vary by task type τ_i)
```

**Innovation**: Simultaneously optimizes delay, energy, reliability, and cost—addressing the multi-QoS gap identified in fog computing surveys.

---

#### 2️⃣ **Temporal Decay-Weighted Acceptance Probability**

Uses **exponential decay** to prioritize recent interaction history over old data.

**Before** (Standard DL-MATCH):
```
π_ij = σ(θ^T × x_ij)  // All history weighted equally
```

**After** (AC-DL-MATCH):
```
π_ij = σ(α × U_ij + β × π_ij^history × e^(-λ × Δt_ij) + γ × (q_ij^avail / q_ij^total))

Where:
- e^(-λ × Δt_ij) = Exponential temporal decay (λ ≈ 0.1)
- Δt_ij = Time elapsed since last interaction
- q_ij^avail / q_ij^total = Queue availability ratio
```

**Innovation**: Solves the "matching with dynamics" open problem where fog node performance changes over time (degradation, traffic spikes). Recent good/bad experiences matter more than historical average.

**Example**: If Fog Node A was reliable 6 months ago but recently crashed 3 times, the system prioritizes recent failures over old success.

---

#### 3️⃣ **Federated Multi-SDN Deployment (East-West WAN Migration)**

Tasks isolate utility computations strictly behind native **SDN Control Planes**. If Local Network Domains physically exhaust, Tasks migrate cross-domain organically avoiding legacy graph complexity scaling limitations. Complexity drastically drops from **O(M×N)** to **O(M×local_fogs)** where **local_fogs << N**.

**Formula**:
```
U_ij^cross-domain = { U_ij^init                           if F_j ∈ Domain_local
                    { U_ij^init × (1.0 - PENALTY_CROSS)   if F_j ∈ Domain_neighbor

Where:
- PENALTY_CROSS = 0.20 (20% standard drop reflecting empirical WAN edge interconnect delay)
```

**The East-West Scaling Paradox (Novel Finding)**:
Through rigorous Monte Carlo simulation, we proved that combining decentralized DRL with strict utility floors (`U >= 0.30`) causes fatal **Synchronization Traps**. When local nodes saturate, strict thresholds artificially reject viable neighbor nodes, forcing catastrophic cloud escalations. AC-DL-MATCH solves this by implementing *unconstrained relative scoring*, allowing organic East-West migration during "Thundering Herd" events and boosting acceptance rates by 30%.

**Scalability Impact**:
- **Standard DL-MATCH**: 10,000 tasks × 1,000 global fog nodes = **10,000,000 utility calculations**
- **AC-DL-MATCH**: 10,000 tasks × 10 local SDN bounds = **100,000 calculations** (100× faster federation speedup)

**Hardware Feasibility**: Makes system lightweight enough for ESP32/Raspberry Pi:
- **AC-DL-MATCH**: SDN evaluates paths remotely, Edge triggers lightweight matrix operations.
- **Deep Learning (DRL)**: 10,000+ iterations local processing causes memory overload on microcontrollers.

#### 4️⃣ **High Fidelity Simulation Engine**

Fully decouples mathematical convergence (DRL and Meta-Heuristics) using a deterministic Monte-Carlo execution thread. 
No third-party Java-dependent simulation engines bounding memory allocation limits.

**Core Simulation Mechanics:**
- **Poisson Burst Task Arrivals:** Models real-world IoT traffic using Poisson distributions ($\lambda=0.8$) to represent bursty, unpredictable network alarms (overcoming legacy Constant Bit Rate models).
- **Heterogeneous Application Cycling:** Edge Nodes dynamically cycle their application priorities on the fly to emulate real-world, multi-tasking sensor realities (e.g., an autonomous vehicle switching from caching maps to emergency braking).
- **Strict Benchmarking Constraints:** All baseline algorithmic bounds operate identically on a mathematically seeded topology under physical infrastructure elasticity limiting (`MAX_SDN_FOGS`), ensuring algorithm results are isolated from hardware rigging.
- **Context-Mapped ML Architectures:** Appends Edge subjective weights explicitly inside Deep Reinforcement Learning state spaces `(4 × max_sdn_fogs) + 4` ensuring identical contextual knowledge among baselines.

---

#### 5️⃣ **Decoupled Subjective QoS Mapping ($O(1)$ SLA Bounding)**

Centralized computing models calculate cost/delay from the physical Server's uniform perspective. Our architecture decouples **Subjective QoS** from **Objective Hardware states**.

Rather than cross-communicating gradients, Edge nodes natively execute an $O(1)$ Decentralized SLA-Bounding Protocol:
$$U_{norm} = \max(0, 1 - \frac{x}{X_{max}})$$
This maps disparate magnitudes universally into an equivalent $\in [0, 1]$ dimension space exactly upon resource discovery—neutralizing dimensional collapse instantly without synchronization overhead.

---

#### 5️⃣ **Distributed Infrastructure Elasticity (Scale-Out)**

**FIRST WORK** to integrate matching algorithm outcomes with infrastructure scaling decisions.

##### **Scale-Out Policy** (Add Fog Nodes)

Trigger Condition:
```
IF (ρ_reject > θ_reject):
    Add New Fog Node
```

**Meaning**:
- Too many tasks rejected locally (`ρ_reject > θ_reject`)
- **Action**: Provision new fog node to increase local capacity

**Thresholds** (typical values):
- `θ_reject` = 15-20% (rejection rate threshold)

##### **Scale-In Policy** (Remove Fog Nodes)

Trigger Condition:
```
IF (μ_j < θ_util) for duration > θ_time:
    Remove Fog Node F_j
```

**Formula**:
```
μ_j = (1/W) × Σ(# Active Tasks / Q_j) over sliding window W

Where:
- μ_j = Average utilization over time window
- Q_j = Total capacity (queue size) of fog node
- W = Sliding window (e.g., last 10 time slots)
```

**Thresholds** (typical values):
- `θ_util` = 20-30% (utilization threshold)
- `θ_time` = 5-10 time slots (sustained low usage)

**Energy Savings**: Removes idle fog nodes while maintaining capacity headroom to handle sudden spikes.

---

#### 6️⃣ **Distributed Architecture**

Unlike centralized cloud or hierarchical fog-cloud, AC-DL-MATCH operates **fully distributed**:

- **Tasks**: Compute utility locally, make independent offloading decisions
- **Fog Nodes**: Accept/reject tasks based on local state, no central coordinator
- **SDN Controllers**: Manage network topology, facilitate multi-domain routing (per domain)

**Advantage**: No single point of failure, scales horizontally.

---

## 📊 Expected Outcomes

### Performance Improvements
- **15,000+ System Utility Ceiling** shattered (outperforming all baseline heuristics).
- **15-25% reduction** in average task latency (vs. static offloading)
- **30-40% reduction** in energy consumption (vs. always-on infrastructure)
- **>94% Task Acceptance Rate** under high-stress federated loads.
- **50-60% faster** decision-making (vs. centralized Deep Reinforcement Learning)

### Practical Feasibility
- Lightweight enough for **ESP32/Raspberry Pi** class IoT devices
- Scales to **10,000+ IoT devices** (vs. 1,000 limit for ILP)
- Real-time response: **<100ms decision latency** (vs. seconds for DRL)

### Novel Contributions
1. **First integration** of matching theory with dynamic infrastructure elasticity
2. **Context-aware** multi-objective optimization (task-type adaptive)
3. **Temporal decay** weighting for dynamic fog node performance tracking
4. **k-hop locality** for scalable distributed computation
5. **Multi-domain** cross-provider task migration

---

## 🎓 Domain Knowledge: Fog Computing Fundamentals

### What is Fog Computing?

**Cloud Computing**: Centralized data centers (high latency, 100-500ms)  
**Edge Computing**: Computation at device level (limited resources)  
**Fog Computing**: Intermediate layer between cloud and edge (low latency, 10-50ms)

```
[IoT Devices] ←→ [Fog Nodes (close by)] ←→ [Cloud (far away)]
   Sensors          Mini-servers               Data centers
   Actuators        Low latency                High compute
```

**Why Fog?**
- **Latency-critical apps**: Autonomous vehicles, AR/VR, industrial automation
- **Bandwidth savings**: Process data locally, send only aggregates to cloud
- **Privacy**: Sensitive data (healthcare) stays close to source

---

### Task Offloading Explained

**Problem**: IoT devices (sensors, cameras) are weak—can't handle heavy computation.

**Solution**: "Offload" (send) computational tasks to nearby powerful fog nodes.

**Example**:
1. **Surveillance Camera** detects motion
2. **Offloads** video frame to fog node for AI object detection
3. **Fog node** identifies "Person with weapon"
4. **Returns** alert to camera in <50ms

**Challenge**: Which fog node should the camera use? (That's what AC-DL-MATCH solves!)

---

### SDN (Software-Defined Networking)

Traditional networks: Routers/switches make independent decisions (distributed, hard to control).

**SDN**: Centralized controller programs network behavior.

```
┌─────────────────────────────────┐
│   SDN Controller (Brain)        │  ← Centralized logic
│   - Topology management         │
│   - Routing decisions           │
└────────┬───────────┬────────────┘
         │           │
    ┌────▼───┐  ┌───▼─────┐
    │ Switch │  │ Switch  │        ← Dumb forwarding
    └────────┘  └─────────┘
```

**In AC-DL-MATCH**: SDN controllers manage network topology, enable multi-domain routing (tasks jumping between fog provider networks).

---

### Matching Theory Basics

**Classic Problem**: Stable marriage problem (match men to women based on preferences).

**In Fog Computing**: Match **Tasks** to **Fog Nodes** based on utility.

**Key Concept - Utility Function**:
```
Utility = "How good is Fog Node j for Task i?"

Higher utility = Better match
```

**AC-DL-MATCH Decision**:
1. Task calculates utility for nearby fog nodes
2. Proposes to fog node with highest utility
3. Fog node accepts/rejects based on capacity and acceptance probability
4. If rejected, task tries next best fog node

---

## 🔬 Research Methodology

### Implementation Tools

**Simulation Platform**: Custom Python Orchestration Engine
- Natively models IoT devices, fog nodes, and baseline computations without JVM overhead.
- Benchmarks DRL (PyTorch) and Meta-Heuristics (PySwarms) internally.
- Outputs reproducible Plotly HTML/PNG traces iteratively.

**Algorithm Implementation**: Python 3.10+
- PyTorch (DQN optimization)
- Numpy (Mathematical Arrays and Aggregation)
- Scikit (Distribution tracking)

**Data Collection**:
- Task arrival patterns (Poisson distribution)
- Fog node capacities (realistic configurations)
- Network topology (geographical distances, hop counts)

---

### Experimental Setup

#### System Components

**1. Task Nodes (T)**
```
T_i = (S_i, χ_i, τ_i)

Where:
- S_i = Task size (MB)
- χ_i = Computational complexity (CPU cycles)
- τ_i = Task type (delay-sensitive / energy-critical / mission-critical)
```

**Example Tasks**:
- Autonomous Vehicle Path Planning: S=2MB, χ=10^9 cycles, τ=delay-sensitive
- Health Monitor ECG Analysis: S=0.5MB, χ=5×10^8 cycles, τ=mission-critical
- Weather Sensor Data Aggregation: S=0.1MB, χ=10^7 cycles, τ=energy-critical

**2. Fog Nodes (F)**
```
F_j = (f_j, mem_j, Q_j, C_j, R_j)

Where:
- f_j = CPU frequency (GHz)
- mem_j = Memory capacity (GB)
- Q_j = Queue capacity (# concurrent tasks)
- C_j = Cost per bit ($/MB)
- R_j = Reliability score (0-1)
```

**Example Fog Nodes**:
- Small Edge Server: f=2GHz, mem=4GB, Q=10, C=$0.01/MB, R=0.85
- Medium Fog Node: f=3.5GHz, mem=16GB, Q=50, C=$0.02/MB, R=0.95
- High-Performance Fog: f=5GHz, mem=64GB, Q=200, C=$0.05/MB, R=0.99

**3. SDN Controllers (S)**
```
S_k = {F_k, G_k(F_k, E_k)}

Where:
- F_k = Set of fog nodes in domain k
- G_k = Network topology graph (nodes + edges)
- E_k = Network connections with (data_rate, propagation_delay, hop_count)
```

**4. Network Topology**

Based on **geographical proximity** with **maximum reachable hop distance (D_max)**.

**Example**:
```
Domain 1 (City A):
- Fog Nodes: F1, F2, F3
- Connections: F1-F2 (1 hop), F2-F3 (1 hop), F1-F3 (2 hops)

Domain 2 (City B):
- Fog Nodes: F4, F5, F6
- Connections: F4-F5 (1 hop), F5-F6 (1 hop), F4-F6 (2 hops)

Cross-Domain:
- F3-F4 (3 hops) → Enables task migration between cities
```

---

### Evaluation Metrics

#### Primary Metrics

**1. Average Task Latency**
```
Latency = Propagation_Delay + Transmission_Delay + Execution_Delay

Target: <50ms for delay-sensitive tasks
```

**2. Energy Consumption**
```
Energy = P_transmission × (Task_Size / Data_Rate) + P_idle × Execution_Time

Target: 30-40% reduction vs. always-on fog nodes
```

**3. Task Acceptance Rate**
```
Acceptance_Rate = (# Accepted Tasks) / (# Total Task Arrivals) × 100%

Target: >85% acceptance rate under normal load
```

**4. Infrastructure Utilization**
```
Utilization = (# Active Fog Nodes) / (# Total Provisioned Nodes) × 100%

Target: 60-80% (balance between capacity and energy savings)
```

#### Comparative Baselines

Compare AC-DL-MATCH against:

1. **Static Offloading**: Fixed task-to-fog mapping (no adaptation)
2. **Standard DL-MATCH**: Original matching theory (no temporal decay, no k-hop, no elasticity)
3. **Deep Reinforcement Learning (DRL)**: State-of-the-art ML approach (DDPG, A3C)
4. **Greedy Local Offloading**: Always select nearest fog node
5. **Random Offloading**: Random fog node selection

---

### Simulation Scenarios

#### Scenario 1: Traffic Spike (Stress Test)

**Setup**:
- Normal load: 500 tasks/minute
- Sudden spike: 2000 tasks/minute (4× increase)
- Duration: 10 minutes

**Evaluation**:
- How quickly does scale-out trigger?
- Task acceptance rate during spike?
- Latency increase compared to normal?

**Expected Result**: AC-DL-MATCH maintains >80% acceptance vs. <50% for static systems.

---

#### Scenario 2: Fog Node Failure

**Setup**:
- 10 fog nodes operational
- At t=30min, kill 2 high-capacity fog nodes
- Observe system recovery

**Evaluation**:
- Task re-routing latency?
- Cross-domain migration rate?
- System stabilization time?

**Expected Result**: <5 seconds to detect failure and re-route tasks.

---

#### Scenario 3: Heterogeneous Task Mix

**Setup**:
- 40% delay-sensitive (autonomous vehicles)
- 30% energy-critical (battery sensors)
- 30% mission-critical (healthcare)

**Evaluation**:
- Does context-aware utility favor correct fog nodes?
- Energy savings for battery-powered tasks?
- Latency for delay-sensitive tasks?

**Expected Result**: 20% better QoS match vs. single-objective optimization.

---

#### Scenario 4: Multi-Domain Mobility

**Setup**:
- Task originates in Domain A
- Moves geographically toward Domain B (simulated mobility)
- Observe cross-domain handoff

**Evaluation**:
- Handoff latency (task migration time)?
- Service continuity (no dropped tasks)?
- Cross-domain success rate?

**Expected Result**: <100ms handoff latency, 0% task drop rate.

---

## 🚧 Known Limitations & Future Work

### Current Bottlenecks

**1. Assuming Static IoT Devices**
- **Current**: IoT devices don't move
- **Future**: Support mobility (vehicles, drones, wearables)
- **Challenge**: Frequent re-matching as devices move

**2. No Security Considerations**
- **Current**: Trust all fog nodes, no authentication
- **Future**: Encrypted task offloading, fog node reputation system
- **Challenge**: Security overhead impacts latency

**3. No Real-World Deployment/Testing**
- **Current**: Pure simulation (Python/Plotly)
- **Future**: Deploy on real Raspberry Pi + ESP32 testbed
- **Challenge**: Hardware constraints, network unpredictability

---

## 📚 Key References

1. **DL-MATCH (2025)**: "Distributed Learning-Based Matching for Task Offloading in Fog Computing"
2. **Matching Theory Survey (2022)**: "Matching Theory for Wireless Networks: A Contemporary Survey"
3. **Fog Computing Survey (2018)**: "Fog Computing: A Comprehensive Survey on Concepts, Architectures, and Applications"
4. **DRL Analysis (2025)**: "Deep Reinforcement Learning for Task Offloading: Challenges and Opportunities"

---

## 🎯 Target Applications

### 1. **Smart Cities**
- Traffic management (real-time route optimization)
- Public safety (surveillance video analytics)
- Smart lighting (adaptive control based on crowd density)

### 2. **Industrial Automation**
- Predictive maintenance (vibration analysis on factory machines)
- Quality control (real-time defect detection on assembly lines)
- Collaborative robots (low-latency coordination)

### 3. **Healthcare**
- Remote patient monitoring (ECG, glucose, blood pressure)
- Fall detection for elderly care
- Emergency response (ambulance route optimization)

### 4. **Agriculture**
- Precision farming (soil moisture, crop health monitoring)
- Autonomous tractors/harvesters
- Livestock tracking

---

## 📖 How to Use This Repository

### For Researchers

1. **Understanding the Algorithm**: Read ARCHITECTURE.md for detailed mathematical formulations
2. **Reproducing Results**: Follow simulation setup in `/matching/` folder
3. **Extending the Work**: Modify utility functions, add new policies in `/matching/algorithms/`

### For Developers

1. **Setting Up Python Env**: Install `requirements.txt` from `/matching/`
2. **Running Experiments**: Use `/matching/main_simulation.py --tests`
3. **Analyzing Results**: Benchmark graphs available in `/matching/results/`

### For Students

1. **Learning Fog Computing**: Read "Domain Knowledge" section above
2. **Understanding Matching Theory**: See `/docs/MATCHING_THEORY_PRIMER.md`
3. **Exploring Code**: Start with `/matching/algorithms/get_utility.py`

---

## 🤝 Contributing

This is a research project. Contributions welcome for:
- Bug fixes in simulation code
- Additional baseline comparisons
- Real-world testbed deployment scripts
- Visualization improvements

---

## 📧 Contact

**Authors**:
- Pratham Aggarwal (22JE0718)

**Institution**: Indian Institute of Technology (Indian School of Mines) Dhanbad
**Advisor**: [Advisor Name]

---

## 📄 License

This project is for academic research purposes. Code released under MIT License.

---

**Last Updated**: April 2026  
**Project Status**: Simulation Completed (Paper Finalization Phase)
