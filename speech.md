# Kitsune Video Demo Script

## 0:00-0:25 - The Problem

**On screen:** Open Kitsune on the `Climate Control Center` page.

**Speech:**

Schools want to reduce their environmental impact, but their carbon footprint is often hidden across heating, lighting, buildings, transport, and waste. Even when a school has a sustainability budget, it is difficult to know which investment should come first. Solar panels may produce a large reduction, but insulation or LED lighting might deliver better results for a specific budget and season.

Kitsune helps schools compare these decisions before spending real money.

## 0:25-0:50 - The Solution

**On screen:** Scroll through the custom environment controls.

**Speech:**

Kitsune is an AI-powered school climate decision-support system. It uses a custom multi-agent reinforcement learning environment to test sustainability strategies under limited annual budgets, changing weather, operational demand, and multi-year constraints.

Instead of returning one generic climate recommendation, Kitsune simulates how different decisions affect emissions, costs, staff capacity, and the options available later.

## 0:50-1:20 - Custom Environment

**On screen:** Change the episode count, weeks per episode, sustainability budget, weather intensity, and starting calendar week.

**Speech:**

The environment can be customized for different scenarios. We can choose up to one thousand episodes, simulate between four weeks and five school years per episode, set the district's annual sustainability budget, choose when the simulation starts, and control demand, weather, human error, and system shocks.

Every simulated school year contains fifty-two weeks and seven generated weather days per week. Annual budgets renew at the beginning of each new school year, allowing the policy to learn long-term planning instead of optimizing only one isolated decision.

## 1:20-1:45 - AI Architecture

**On screen:** Show the action list or briefly switch to the code/notebook architecture.

**Speech:**

Four agents control four school hubs. Each agent observes fifty-three features, including budget health, infrastructure emissions, renewable readiness, retrofit backlog, weather pressure, fatigue, and shared city conditions.

Each agent can choose between thirteen actions. Seven are physical sustainability investments: solar panels, LED lighting, insulation, smart thermostats, HVAC upgrades, recycling programs, and electric buses.

## 1:45-2:20 - Run the Simulation

**On screen:** Click `Run custom simulation` and show the first climate metrics.

**Speech:**

When we run the scenario, the policy evaluates sequences of investments instead of selecting the action with the largest immediate reduction. The reward encourages lower carbon emissions and efficient budget use, while penalties prevent overspending, staff burnout, inequity, and unsafe operational shortcuts.

At the top, we can compare the initial carbon footprint, the current footprint, the reduction achieved, the best solution found, and how much of the sustainability budget remains.

## 2:20-2:55 - Explainability and Decision Value

**On screen:** Show carbon progress, Budget Allocation, Action Frequency, ROI, and School Digital Twin.

**Speech:**

A final reward score would not be useful by itself, so Kitsune explains the policy through decision-focused metrics.

The budget allocation chart shows where resources were spent. Action frequency shows which interventions the policy repeatedly preferred. The ROI table compares estimated CO2 reduction with cost, and the school digital twin shows which systems remain high-emission, moderate, or optimized.

This helps a school understand not only what the model selected, but why that strategy appears valuable under the chosen constraints.

## 2:55-3:20 - Responsible AI

**On screen:** Keep the ROI and budget information visible.

**Speech:**

The main risk is over-reliance. Our training scenarios, intervention costs, and carbon reductions are synthetic prototype estimates. They must not be treated as guaranteed real-world savings.

To reduce this risk, Kitsune exposes the assumptions, estimated costs, reductions, ROI, remaining budget, and operational constraints. It does not hide uncertainty behind one recommendation, and it never purchases equipment automatically.

## 3:20-3:40 - Human-in-the-Loop

**On screen:** Show the dashboard summary or responsible-AI section.

**Speech:**

Kitsune does not approve the final school investment plan. A school administrator or facilities manager must verify real vendor quotes, building safety, procurement rules, local regulations, maintenance capacity, and community priorities before committing money.

AI supports the decision, but a qualified human remains responsible for implementation.

## 3:40-4:00 - Closing

**On screen:** Return to the Climate Control Center overview with the best carbon result visible.

**Speech:**

Kitsune turns a school's hidden footprint into a transparent climate-action experiment. It lets schools test budgets, weather, and investment strategies, understand the trade-offs, and move from general climate awareness toward evidence-based local action.

That is how Kitsune makes climate action local, understandable, and usable.
