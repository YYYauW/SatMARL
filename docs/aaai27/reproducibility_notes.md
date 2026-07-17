# Reproducibility checklist notes

- Code, environment configuration, checkpoints, and per-episode metrics are stored under the project root.
- Main comparison launcher: `scripts/start_aaai_rl_suite.ps1`.
- Algorithms: OASIS, IPPO, MAPPO, QMIX, and parameter-shared Double DQN.
- Training scenarios: global-random, unique seed per episode, base seed 401.
- Held-out scenarios: seeds 9001--9050, shared by every method.
- Main environment: 16 satellites, 4 planes, 768 tasks, 240 steps, 30 seconds per planning step, 5 seconds per point acquisition, 45-degree FOV.
- OASIS update threshold: 512 effective decision opportunities, maximum four buffered episodes.
- Hardware for the active run: one CUDA GPU; exact model name and elapsed time must be copied into the final paper after completion.
- Randomness: final submission requires at least three independent training seeds; five are stated in the draft and must be run or revised before submission.
- Statistical analysis: paired differences on identical evaluation seeds, 95% confidence intervals, and multiple-training-seed mean plus standard deviation.
- No human-subject data or sensitive personal data are used.
- Known simulator limitations are listed in the paper's Limitations and Scope section.
