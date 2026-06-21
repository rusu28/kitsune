# Kitsune

Kitsune is a school climate decision-support prototype built for **Make Climate Action Local and Real - My School's Hidden Footprint**. It uses a custom multi-agent reinforcement learning environment to compare sustainability investments under annual budgets, changing weather, operational pressure, and multi-year constraints.

The system does not approve purchases or claim guaranteed savings. It helps a school compare simulated strategies before a human decision-maker reviews real costs, regulations, safety, and local priorities.

## What Kitsune Does

Kitsune models four school hubs and lets each policy allocate resources across seven sustainability initiatives:

- solar panels;
- LED lighting;
- building insulation;
- smart thermostats;
- HVAC upgrades;
- recycling programs;
- electric buses.

The Streamlit control center lets users configure:

- 1 to 1,000 simulation episodes;
- 4 to 260 weeks per episode;
- annual support and sustainability budgets;
- starting calendar week;
- demand and weather intensity;
- human-error and system-shock rates.

The dashboard reports carbon footprint, reduction achieved, budget allocation, remaining budget, action frequency, estimated ROI, daily weather, policy reward, safety counter-metrics, and school digital-twin status.

## Quick Start on Windows

Requirements:

- Windows 10 or 11;
- Python 3.10 or newer;
- internet access for the first dependency/model installation.

Double-click:

```text
run_kitsune.bat
```

Then choose:

```text
[1] Install dependencies and cache local Qwen
[2] Open Streamlit
[3] Exit
```

The launcher selects the first free port from `8501` upward and prints both links:

```text
Local link:   http://localhost:8501
Network link: http://YOUR_LOCAL_IP:8501
```

Keep the launcher window open while the app is running. Press `Ctrl+C` in that window to stop Streamlit.

## Recommended Requirement for Better Results: Qwen

Kitsune can run without Qwen by using its guarded deterministic fallback. For the complete plain-language experience, install and cache the free local model **Qwen/Qwen2.5-0.5B-Instruct** through option `1` in `run_kitsune.bat`.

Qwen runs locally through Hugging Face Transformers and PyTorch. It does not select sustainability investments or control the reinforcement learning policy.

To cache it manually:

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-0.5B-Instruct'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"
```

## Manual Installation

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Reinforcement Learning Environment

The custom Gymnasium environment has:

- four agents, one for each school hub;
- `53` observed features per agent;
- `13` continuous actions per agent;
- a 52-week school calendar;
- seven generated weather days per simulated week;
- annual budgets that renew each school year;
- stochastic demand, human error, and system shocks;
- multi-year episodes;
- deterministic seeds for reproducible comparisons.

Each policy balances carbon reduction and budget efficiency against fatigue, inequity, unresolved need, wrong guidance, and missed urgent cases. The additional counter-metrics prevent a policy from looking successful by optimizing only one easy score.

## Use the Environment in Python

```python
from school_help_env import CrisisToActionCityEnv

env = CrisisToActionCityEnv(
    seed=42,
    episode_weeks=104,
    district_support_budget=1_500_000.0,
    district_eco_budget=500_000.0,
    weather_intensity=1.25,
    human_error_rate=1.10,
    shock_rate=1.20,
)

state, info = env.reset(seed=42)
agent_states = env.get_agent_observations()

print(env.agent_state_dim)   # 53
print(env.agent_action_dim)  # 13
```

For PPO training, open `ppo_agent.ipynb` and construct each agent with:

```python
agent = Agent(env.agent_state_dim, env.agent_action_dim)
```

## Architecture

```text
School state + budgets + weather + infrastructure
                         |
                         v
        Multi-agent PPO policy / scenario runner
                         |
                         v
       13 resource-allocation actions per school
                         |
                         v
 Carbon, budget, ROI, safety, equity, and fatigue metrics
                         |
                         v
       Streamlit climate decision-support dashboard
```

The built-in Streamlit runner uses a fast fixed policy for immediate scenario comparisons. PPO training uses the same observation and action spaces in the notebook.

## Responsible AI

The core environment uses seeded synthetic data. It does not use live utility bills, vendor quotes, regional carbon factors, or real student records.

Main risk: users could treat simulated CO2 savings as guaranteed real-world outcomes.

Guardrail: Kitsune exposes estimated costs, reductions, ROI, budget use, and constraints, labels the results as scenario estimates, and never approves or purchases an investment. A school administrator or facilities manager must verify real-world evidence before implementation.

## Project Files

```text
app.py                     Streamlit interface and climate dashboard
school_help_env.py         Gymnasium environment, actions, rewards, Qwen layer
ppo_agent.ipynb            PPO training notebook
city_simulator_pygame.py   Optional Pygame visualization
run_kitsune.bat            Windows installer and launcher
requirements.txt           Python dependencies
```

## Technology

Python, PyTorch, Gymnasium, NumPy, Streamlit, Pygame, Jupyter, Hugging Face Transformers, PPO, and Qwen2.5.

## Current Limitations

- Climate and cost values are synthetic prototype estimates.
- The dashboard does not query live regional or utility data.
- The fixed dashboard policy is for rapid comparison, not a replacement for trained PPO evaluation.
- Real deployments require building audits, verified emission factors, vendor quotes, and human approval.
