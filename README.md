# Mario Kart DS - RL Agent (DQN/PPO)

## Description 

This project features an autonomous Reinforcement Learning agent developed for playing the popular game **Mario Kart DS** autonomously. By leveraging **Deep Q-Networks (DQN)**, **Convolutional Neural Networks (CNNs)**, and the **Stable-Baselines3 (SB3)** framework, the project trains an agent to navigate race tracks using a sophisticated dual-input approach. The system processes high-level **visual data** (top screen capture) alongside **low-level telemetry** (RAM access) retrieved directly from the **DeSmuME** emulator via the `py-desmume` library.

It was developed as part of the **Autonomous Agents (ΠΛΗ 412, 2025-2026)** course at the **Technical University of Crete**.

---

## Project Structure

```text
root/
├── env/
│   ├── mkds_custom_env.py        # Custom Gymnasium env (DQN training)
│   └── mkds_gym_env.py           # Gymnasium env (SB3 training)
├── envs/
│   └── mkds_env/                 # Python virtual environment
├── outputs/                      # Checkpoints + final trained model (.zip)
├── logs/                         # Training logs (DQN_1, DQN_2, ...)
├── rom/                          # Placeholder for ROM (PUT ROM HERE, MUST BE USA VERSION)
├── src/                          
│   ├── agents/
│   │   └── dqn_agent.py          # Custom DQN implementation
│   └── utils/
│       ├── config.py             # Global configs & hyperparameters
│       ├── visualization.py      # Training & gameplay visualizations
│       ├── wrappers.py           # Observation/action wrappers
│       └── ram_vars_testing.py   # RAM access validation script
├── mkds_boost.dst                # Save-state (race start position)
├── requierments.txt              # Python dependencies
├── train_tf_dqn.py               # Secondary training entry (TF DQN)
├── train_sb3_ppo.py              # Primary training entry (SB3 PPO)
├── demo_sb3.py                   # Run trained model (inference/demo)
└── emu_documentation.txt         # py-desmume API notes & summary
```

## Getting Started

### Installation / Dependencies
This project was developed in a Windows 11 enviroment but should work in Linux/MacOS as well.
* Python version: **3.12.7** 
* Create a new python venv and install the dependancies in `requierments.txt`
* You need to own a **ROM (USA Version)** of the game. Place it inside the `rom/` folder.


### Executing program

You can select between `train_sb3_ppo.py` and `train_tf_dqn.py`

```bash
python train_sb3_ppo.py
```
```bash
python train_tf_dqn.py
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)