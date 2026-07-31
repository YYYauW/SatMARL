# Title

OASIS: Opportunity-Aware Asynchronous Multi-Agent Reinforcement Learning for Constraint-Rich Multi-Satellite Scheduling

# Abstract

Large Earth-observation constellations create a decentralized scheduling problem in which each satellite acts under time windows, orbital visibility, attitude, payload, energy, storage, cooperation, and ground-station constraints. Although multi-agent reinforcement learning (MARL) is attractive for online replanning, conventional fixed-step training is dominated by forced-idle transitions: at most times, most satellites have no feasible decision. This produces inefficient policy updates and incorrect temporal credit assignment when useful decisions occur at irregular intervals. We introduce OASIS, an opportunity-aware asynchronous MARL method that represents each satellite's decision process as a semi-Markov sequence of genuine decision opportunities. OASIS combines per-agent opportunity filtering, duration-corrected generalized advantage estimation, branch-balanced policy updates, and centralized training with decentralized execution. We evaluate OASIS in a constraint-rich multi-plane low-Earth-orbit constellation simulator with five-second point observations, a 120-minute horizon, heterogeneous payloads, cooperative observations, and segmented downlink. The controlled study compares OASIS with IPPO, MAPPO, QMIX, and parameter-shared DQN under identical training and held-out scenarios, with heuristic schedulers retained only as diagnostic lower bounds. The study tests whether explicitly modeling sparse decision opportunities improves sample efficiency, task utility, and coordination without violating mission constraints.

# OpenReview topics

- Primary: MAS: Multiagent Learning
- Secondary: MAS: Multiagent Planning & Coordination
- Secondary: PRS: Learning for Planning & Scheduling
- Secondary: ML: Reinforcement, Imitation & Inverse RL
- Optional: PRS: Planning under Uncertainty & Markov Models

# Keywords

multi-agent reinforcement learning; asynchronous decision making; semi-Markov decision process; satellite constellation scheduling; decentralized planning; constrained scheduling
