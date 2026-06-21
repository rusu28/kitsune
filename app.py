from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from school_help_env import (
    CASE_LIBRARY,
    DEFAULT_SCHOOL_SETTINGS,
    SCHOOL_NAMES,
    CrisisToActionCityEnv,
    SchoolHelpConfig,
    ask_city_assistant,
    warm_local_qwen,
)


st.set_page_config(
    page_title="Kitsune",
    page_icon="K",
    layout="wide",
)


PROJECT_NAME = "Kitsune"
TAGLINE = "Local AI that turns confusing support notices into safe, clear next steps."
DEFAULT_ENV_CONFIG = SchoolHelpConfig()
DEFAULT_DISTRICT_SUPPORT_BUDGET = int(
    sum(
        DEFAULT_ENV_CONFIG.annual_support_budget_base
        + DEFAULT_ENV_CONFIG.annual_support_budget_scale * school["budget"]
        for school in DEFAULT_SCHOOL_SETTINGS
    )
)
DEFAULT_DISTRICT_ECO_BUDGET = int(
    sum(70000.0 + 65000.0 * school["budget"] for school in DEFAULT_SCHOOL_SETTINGS)
)
MARIA_NOTICE = (
    "I got a school notice about temporary housing support, eligibility papers, proof of address, "
    "and a deadline this Friday. My family may lose our place soon, and I do not understand what "
    "I need to do first or who can safely check the form."
)

JUDGE_SCENARIOS = (
    {
        "label": "Maria housing notice",
        "case_type": "housing_support_notice",
        "message": MARIA_NOTICE,
    },
    {
        "label": "Food assistance form",
        "case_type": "food_assistance_form",
        "message": (
            "My family needs food support, but the form uses terms like income verification and "
            "household eligibility. I do not know what proof is accepted or what to do first."
        ),
    },
    {
        "label": "Counseling support page",
        "case_type": "mental_health_support",
        "message": (
            "I feel overwhelmed and found a long school page about counseling. I cannot tell "
            "who to contact first or what information stays private."
        ),
    },
)

ABOUT_PROJECT_COPY = """## Inspiration
We started with Maria, a parent under deadline stress. Housing support may exist, but a long school notice, unfamiliar eligibility language, and missing-document anxiety can keep her from acting in time. A web search can find links; it does not turn Maria's exact message into a safe first action or explain when a person must take over.

## What it does
Maria pastes the notice into Kitsune. A free local Qwen model rewrites it in plain language. Deterministic rules provide case routing, urgency and source-confidence signals, a checklist, source-check prompts, explicit AI limits, and a human-review handoff. Before Kitsune, Maria has uncertainty and no first step. After Kitsune, she has an actionable plan without the AI deciding eligibility. A multi-agent PPO simulator stress-tests how four support hubs allocate staff and annual budget across a 52-week school year with human error and weather disruptions.

## How we built it
The app uses Python and Streamlit. Qwen/Qwen2.5-0.5B-Instruct runs locally through Hugging Face Transformers and PyTorch. Guarded Python logic handles routing, scoring, checklists, prompt-injection filtering, output validation, and fallback responses. Gymnasium, NumPy, and PPO power the support-hub simulation; Pygame provides a second viewer.

## Challenges we ran into
The small local model initially repeated unsafe instructions and sometimes treated unverified claims as facts. We separated generation from decisions, sanitized untrusted input, validated high-risk outputs, and added deterministic fallbacks. We also redesigned the reward so agents could not earn high scores from trust metrics while helping too few people or exhausting staff and budget.

## Accomplishments we're proud of
Kitsune runs without a paid inference API, demonstrates a complete input-to-action flow, exposes its uncertainty and limits, and measures both help delivered and safety failures. The same project shows the user experience first and then stress-tests how that workflow behaves at city scale.

## What we learned
Responsible AI is not a warning added at the end. For this use case, the strongest design is hybrid: generation improves clarity, while explicit rules and humans retain control of consequential decisions. We also learned that impact metrics need counter-metrics such as wrong guidance, missed urgency, fatigue, inequity, and overspending.

## What's next
Next we would test with students, parents, and school support staff; add multilingual rewriting; connect only to verified official service directories with citations; measure comprehension before and after; and deploy a privacy-reviewed pilot with clear escalation procedures."""

DATA_SOURCES_DISCLOSURE = (
    "All demo cases are synthetic. We manually created scenarios from categories named in Challenge Brief 1: "
    "housing-support notices, food-assistance forms, counseling pages, academic-support requests, and "
    "unverified service claims. Simulator demand, human-error events, weather, budgets, and outcomes are "
    "generated with seeded NumPy distributions. No real student, medical, housing, or benefits records are "
    "used, and no external service-directory API is queried. Qwen receives only the text entered for the "
    "current response; Kitsune does not intentionally persist it. The challenge brief informed scenario "
    "categories but was not used as model-training data."
)

BUILT_WITH_TAGS = (
    "Python, Streamlit, Gymnasium, NumPy, PyTorch, Hugging Face Transformers, "
    "Qwen2.5, PPO, Pygame, Jupyter"
)

DEVPOST_REQUIRED_FIELDS = (
    {
        "field": "Elevator Pitch",
        "check": "Specific user + problem + outcome in one tight paragraph.",
        "copy": (
            "Kitsune helps Maria, a parent under deadline stress, turn a confusing school "
            "support notice into plain language, safe next steps, confidence labels, source checks, "
            "and a human-review route. The prototype also simulates how school support hubs can scale "
            "that help without rewarding unsafe shortcuts."
        ),
    },
    {
        "field": "About the Project",
        "check": "Real user, before/after, working prototype, impact metrics.",
        "copy": ABOUT_PROJECT_COPY,
    },
    {
        "field": "AI Architecture Explanation",
        "check": "Clear input -> AI process -> output flow, with named AI capabilities.",
        "copy": (
            "Input: Maria's confusing school support notice. Processing: deterministic case routing and "
            "urgency/source-confidence rules structure the case; local Qwen2.5-0.5B rewrites it in plain "
            "language; guarded templates produce next steps, source-check prompts, and a human-review flag. "
            "A multi-agent PPO simulator stress-tests staffing and annual-budget policies. Output: explanation, "
            "checklist, confidence, AI limits, human handoff, and impact metrics."
        ),
    },
    {
        "field": "Human-in-the-Loop Decision",
        "check": "Specific handoff point and who takes over.",
        "copy": (
            "Kitsune does not decide whether Maria is eligible for housing support. When eligibility, "
            "missing documents, high urgency, or low confidence affects the case, Kitsune routes her to a "
            "school housing liaison or support staff member. A human must remain in control because the "
            "decision depends on verified records, current policy, and personal context the AI cannot safely confirm."
        ),
    },
    {
        "field": "Responsible AI Guardrail",
        "check": "Real risk + concrete mitigation, not generic ethics language.",
        "copy": (
            "Risk: local Qwen could turn an unverified or maliciously instructed message into confident but "
            "wrong guidance. Mitigation: generated text never controls decisions. Kitsune sanitizes pasted input, "
            "validates output against deterministic risk rules, and falls back to guarded text; sensitive or "
            "low-confidence cases always surface source checks and a human handoff. Synthetic demo data avoids "
            "exposing personal records."
        ),
    },
)

AI_TOOLS_USED_DISCLOSURE = (
    "Runtime: Python, Streamlit, Gymnasium, NumPy, PyTorch, Hugging Face Transformers, "
    "Qwen/Qwen2.5-0.5B-Instruct, and Pygame. Qwen runs locally/offline and all runtime tools are free; "
    "no paid inference API is used. Codex was used as AI coding assistance for implementation, testing, "
    "wording, and submission polish (availability/cost depends on the user's plan). Demo data is synthetic."
)

STATUS_COLORS = {
    "active": "#5aa7ff",
    "probation": "#f4be4f",
    "recovery": "#50d2dc",
    "model_hub": "#46d284",
    "collapsed": "#ee5f5f",
}


class StreamlitHeuristicAgent:
    def __init__(self, env: CrisisToActionCityEnv) -> None:
        self.local_names = list(env.LOCAL_OBS_NAMES)
        self.action_bias = float(env.config.action_activation_bias)

    def act(self, state: np.ndarray) -> np.ndarray:
        local = {name: float(state[idx]) for idx, name in enumerate(self.local_names)}
        desired = np.array(
            [
                0.30 + 0.58 * local["confusing_documents"] + 0.22 * local["language_barrier"] + 0.10 * local.get("weather_disruption", 0.0),
                0.24 + 0.68 * local["rumor_load"] + 0.26 * local["misinformation_pressure"],
                0.36 + 0.70 * local["unmatched_requests"] + 0.18 * local["service_availability"],
                0.32 + 0.78 * local["urgent_cases"] + 0.22 * local["last_missed_urgent"] + 0.20 * local.get("weather_disruption", 0.0),
                0.34 + 0.76 * local["followup_backlog"] + 0.26 * local["unmatched_requests"] + 0.20 * local.get("transportation_barrier", 0.0),
                0.18 + 0.40 * local["worker_fatigue"] + 0.18 * (1.0 - local["worker_capacity"]) + 0.12 * local.get("attendance_disruption", 0.0),
                0.18 + 0.58 * local.get("renewable_readiness", 0.0) + 0.18 * local.get("building_emissions", 0.0),
                0.14 + 0.70 * local.get("lighting_emissions", 0.0),
                0.16 + 0.62 * local.get("retrofit_backlog", 0.0) + 0.18 * local.get("heating_emissions", 0.0) + 0.25 * local.get("heating_support_need", 0.0),
                0.12 + 0.50 * local.get("heating_emissions", 0.0) + 0.22 * local.get("building_emissions", 0.0) + 0.24 * local.get("cooling_support_need", 0.0),
                0.12 + 0.65 * local.get("heating_emissions", 0.0) + 0.20 * local.get("heating_support_need", 0.0),
                0.12 + 0.68 * local.get("waste_emissions", 0.0),
                0.10 + 0.72 * local.get("transport_emissions", 0.0) + 0.25 * local.get("transportation_barrier", 0.0),
            ],
            dtype=np.float32,
        )
        annual_budget = local.get("annual_budget_left", 1.0)
        desired[:3] *= 0.62 + 0.38 * annual_budget
        desired[3] *= 0.82 + 0.18 * annual_budget
        desired[4:6] *= 0.55 + 0.45 * annual_budget
        desired = np.clip(desired, 0.10, 0.88)
        logits = np.log(desired / (1.0 - desired))
        return (logits + self.action_bias).astype(np.float32)


def main() -> None:
    apply_contest_style()
    st.title(PROJECT_NAME)
    st.caption(TAGLINE)

    page = st.sidebar.radio(
        "View",
        ["Climate Control Center", "Ask for Help", "Support Safety Demo"],
        index=0,
    )

    if page == "Climate Control Center":
        city_simulator_view()
    elif page == "Ask for Help":
        ask_for_help_view()
    else:
        judge_demo_view()


def apply_contest_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        div[data-testid="stMetric"] {
            background: #101826;
            border: 1px solid #22314a;
            border-radius: 8px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricLabel"] {color: #9fb1c7;}
        div[data-testid="stMetricValue"] {color: #f3f7fb; font-size: 1.35rem;}
        .brief-band {
            border-left: 4px solid #4aa3ff;
            background: #0f1724;
            color: #eaf1f8;
            padding: 14px 16px;
            border-radius: 6px;
            margin: 8px 0 14px 0;
        }
        .judge-answer {
            background: #0f1724;
            border: 1px solid #22314a;
            color: #eaf1f8;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 140px;
        }
        .score-item {
            border: 1px solid #d7e0eb;
            background: #f7f9fc;
            color: #172033;
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 88px;
        }
        .score-item b {display: block; color: #315f91; margin-bottom: 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def judge_demo_view() -> None:
    st.header("Judge Demo")
    st.markdown(
        """
        <div class="brief-band">
        <b>User:</b> Maria, a parent under deadline stress.<br>
        <b>Problem:</b> help exists, but the notice is confusing and risky to interpret alone.<br>
        <b>Outcome:</b> plain language, checklist, confidence, human review, then city-scale impact metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_round_one_readiness()
    render_scorecard()
    render_before_after_band()

    st.subheader("1. User-Facing AI")
    selected = st.segmented_control(
        "Scenario",
        [item["label"] for item in JUDGE_SCENARIOS],
        default=JUDGE_SCENARIOS[0]["label"],
    )
    selected_index = [item["label"] for item in JUDGE_SCENARIOS].index(selected)
    selected_scenario = JUDGE_SCENARIOS[selected_index]
    result_key = f"judge_qwen_{selected_scenario['case_type']}"
    if st.button("Run selected case with local Qwen", width="stretch"):
        with st.spinner("Running free local Qwen with safety guardrails..."):
            st.session_state[result_key] = ask_city_assistant(
                selected_scenario["message"],
                case_type=selected_scenario["case_type"],
                use_llm=True,
            )
    selected_result = st.session_state.get(result_key)
    if selected_result is None:
        selected_result = ask_city_assistant(
            selected_scenario["message"],
            case_type=selected_scenario["case_type"],
        )
    render_judge_scenario(selected_scenario, selected_result)

    st.subheader("2. City-Scale Safety Proof")
    col_run, col_note = st.columns([1, 2])
    with col_run:
        run_demo = st.button("Run judge simulation", type="primary", width="stretch")
    with col_note:
        st.caption(
            "The simulation stress-tests scaling without rewarding wrong guidance, missed urgent cases, "
            "staff burnout, or low-help stagnation."
        )

    if run_demo or "judge_history" not in st.session_state:
        history, last_info = run_dashboard_simulation(episodes=8, episode_weeks=8, seed=42)
        st.session_state["judge_history"] = history
        st.session_state["judge_last_info"] = last_info

    render_city_metrics(st.session_state["judge_history"], st.session_state["judge_last_info"], compact=True)

    st.subheader("3. Submission Answers")
    render_devpost_answers()
    render_copy_ready_submission()


def render_before_after_band() -> None:
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown(
            """
            <div class="judge-answer">
            <b>Before Kitsune</b><br>
            Maria has a stressful notice, confusing eligibility language, an approaching deadline,
            and no clear first step.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_after:
        st.markdown(
            """
            <div class="judge-answer">
            <b>After Kitsune</b><br>
            Maria gets a plain-language summary, a checklist, confidence, source checks,
            and a human-review route.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_round_one_readiness() -> None:
    st.subheader("Round 1 Screening Ready")
    st.caption(
        "Devpost screening reads these five required fields first. This demo keeps each one complete, "
        "specific, and aligned with the working prototype."
    )
    cols = st.columns(5)
    for col, item in zip(cols, DEVPOST_REQUIRED_FIELDS):
        col.success(item["field"])
        col.caption(item["check"])


def render_scorecard() -> None:
    cols = st.columns(5)
    score_items = [
        ("Problem 30%", "Maria under deadline stress"),
        ("AI 20%", "Local Qwen + guardrails"),
        ("Design 20%", "Hybrid AI -> safe action"),
        ("Impact 20%", "Helped + safety metrics"),
        ("Responsible 10%", "Human in control"),
    ]
    for col, (label, value) in zip(cols, score_items):
        col.markdown(
            f'<div class="score-item"><b>{label}</b>{value}</div>',
            unsafe_allow_html=True,
        )


def render_judge_scenario(scenario: dict[str, str], result: dict[str, Any]) -> None:
    col_input, col_output, col_safety = st.columns([1.15, 1.4, 1.0])
    with col_input:
        st.markdown("**User Message**")
        st.write(scenario["message"])
        st.caption(f"Detected case: {result['case_type'].replace('_', ' ')}")

    with col_output:
        st.markdown("**Plain Language + Next Steps**")
        st.info(result["summary"])
        if result.get("llm_backend", "guarded_template") != "guarded_template":
            latency = int(result.get("llm_latency_ms", 0))
            st.caption(
                f"Backend: {result.get('llm_backend')} | {latency / 1000:.2f}s | "
                "generation cannot change safety decisions"
            )
        for idx, step in enumerate(result["next_steps"], start=1):
            st.write(f"{idx}. {step}")

    with col_safety:
        st.markdown("**Safety Decision**")
        st.metric("Confidence", result["confidence_label"].upper())
        st.metric("Urgency", f"{result['urgency']:.2f}")
        st.metric("Human review", "YES" if result["human_review_needed"] else "No")
        st.warning(result["safeguard_note"])

    render_source_and_limits(result)


def render_source_and_limits(result: dict[str, Any]) -> None:
    col_sources, col_limits = st.columns(2)
    with col_sources:
        st.markdown("**Source Transparency**")
        for item in result.get("source_checks", []):
            st.write(f"- {item}")
    with col_limits:
        st.markdown("**What AI Does Not Decide**")
        for item in result.get("ai_limits", []):
            st.write(f"- {item}")


def render_devpost_answers() -> None:
    for row_start in range(0, len(DEVPOST_REQUIRED_FIELDS), 2):
        cols = st.columns(2)
        for col, item in zip(cols, DEVPOST_REQUIRED_FIELDS[row_start : row_start + 2]):
            preview = item["copy"]
            if len(preview) > 280:
                preview = preview[:277].rstrip() + "..."
            with col:
                st.markdown(
                    f"""
                    <div class="judge-answer">
                    <b>{item["field"]}</b><br>
                    {preview}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_copy_ready_submission() -> None:
    st.subheader("Results Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training episode", "218")
    col2.metric("Reward / week", "1.09")
    col3.metric("People helped", "0.37")
    col4.metric("Fatigue / budget", "0.21 / 0.75")
    st.caption(
        "Representative PPO run: helped_ep=0.37, helped_last=0.40, wrong=0.04, missed=0.03, "
        "fatigue=0.21, budget=0.75, with a model_hub present. The newer environment also tracks "
        "carbon footprint, carbon reduction, paper forms avoided, travel avoided, and eco budget."
    )

    st.subheader("Copy-Ready Required Devpost Fields")
    for item in DEVPOST_REQUIRED_FIELDS:
        st.text_area(item["field"], value=item["copy"], height=150)

    st.text_area("AI Tools Used", value=AI_TOOLS_USED_DISCLOSURE, height=100)
    st.text_area("Data Sources", value=DATA_SOURCES_DISCLOSURE, height=130)
    st.text_area("Built With tags", value=BUILT_WITH_TAGS, height=80)


def ask_for_help_view() -> None:
    st.header("Ask for Help")
    st.write(
        "Paste a confusing school notice, service claim, support request, or rumor. "
        "The assistant turns it into plain language, a checklist, confidence, and a human-review flag."
    )

    examples = {case["case_type"]: case["message"] for case in CASE_LIBRARY}
    selected_example = st.selectbox("Example scenario", ["Custom"] + list(examples.keys()))
    default_message = MARIA_NOTICE
    if selected_example != "Custom":
        default_message = examples[selected_example]

    user_message = st.text_area("User message", value=default_message, height=150)
    case_type = st.selectbox("Optional case hint", ["Auto-detect"] + list(examples.keys()))
    use_llm = st.toggle(
        "Local Qwen (free + offline)",
        value=True,
        help="Uses the locally cached Qwen2.5 0.5B model. Safety scores and human-review routing stay deterministic.",
    )
    if use_llm:
        warm_local_qwen(background=True)

    if st.button("Generate clear next steps", type="primary"):
        with st.spinner("Creating a safe plain-language explanation..."):
            result = ask_city_assistant(
                user_message,
                case_type=None if case_type == "Auto-detect" else case_type,
                use_llm=use_llm,
            )
        st.session_state["last_assistant_result"] = result

    result = st.session_state.get("last_assistant_result")
    if result:
        render_assistant_result(result)


def render_assistant_result(result: dict[str, Any]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Case type", result["case_type"].replace("_", " "))
    col2.metric("Urgency", f"{result['urgency']:.2f}")
    col3.metric("Clarity", f"{result['clarity_score']:.2f}")
    col4.metric("Source confidence", f"{result['source_confidence']:.2f}")
    col5.metric("Human review", "YES" if result["human_review_needed"] else "No")

    st.subheader("Plain-language summary")
    st.info(result["summary"])
    latency = int(result.get("llm_latency_ms", 0))
    latency_note = f" | {latency / 1000:.2f}s" if latency else ""
    st.caption(f"Assistant backend: {result.get('llm_backend', 'guarded_template')}{latency_note}")
    if result.get("llm_status") == "fallback" and result.get("llm_error"):
        st.warning(result["llm_error"])
    if result.get("prompt_injection_detected"):
        st.warning("Unsafe instructions inside the pasted message were ignored before local generation.")

    st.subheader("Next steps")
    for idx, step in enumerate(result["next_steps"], start=1):
        st.write(f"{idx}. {step}")

    st.subheader("Safeguard")
    st.warning(result["safeguard_note"])
    st.caption(f"Resource hint: {result['resource_hint']}")
    render_source_and_limits(result)

    st.subheader("Before / After")
    col_before, col_after = st.columns(2)
    with col_before:
        st.error(result.get("before_state", "Confusing information and no clear next step."))
    with col_after:
        st.success(result.get("after_state", "Clear summary, checklist, confidence, and human-review route."))


def city_simulator_view() -> None:
    st.header("School Climate Control Center")
    st.write(
        "Compare how four school hubs invest in solar, lighting, insulation, HVAC, recycling, and transport "
        "under annual budgets, seasonal weather, operational demand, and safety constraints."
    )
    st.caption(
        "This fast dashboard runner compares environment scenarios with a fixed policy. "
        "The notebook uses the same state and action spaces for PPO training."
    )

    with st.expander("Custom Environment", expanded=True):
        col_a, col_b, col_c, col_d = st.columns(4)
        episodes = int(
            col_a.number_input("Episodes", min_value=1, max_value=1000, value=12, step=1)
        )
        episode_weeks = int(
            col_b.number_input("Weeks per episode", min_value=4, max_value=260, value=52, step=4)
        )
        seed = int(col_c.number_input("Seed", min_value=0, max_value=999999, value=42, step=1))
        start_calendar_week = int(
            col_d.number_input("Starting calendar week", min_value=1, max_value=52, value=35, step=1)
        ) - 1

        budget_a, budget_b = st.columns(2)
        annual_support_budget = float(
            budget_a.number_input(
                "District annual support budget ($)",
                min_value=100000,
                max_value=10000000,
                value=DEFAULT_DISTRICT_SUPPORT_BUDGET,
                step=50000,
            )
        )
        annual_eco_budget = float(
            budget_b.number_input(
                "District annual sustainability budget ($)",
                min_value=50000,
                max_value=5000000,
                value=DEFAULT_DISTRICT_ECO_BUDGET,
                step=25000,
            )
        )

        pressure_a, pressure_b, pressure_c, pressure_d = st.columns(4)
        demand_multiplier = float(
            pressure_a.slider("Demand pressure", 0.50, 2.00, 1.00, 0.05)
        )
        weather_intensity = float(
            pressure_b.slider("Weather intensity", 0.00, 2.00, 1.00, 0.05)
        )
        human_error_rate = float(
            pressure_c.slider("Human-error rate", 0.00, 2.00, 1.00, 0.05)
        )
        shock_rate = float(
            pressure_d.slider("System-shock rate", 0.00, 2.00, 1.00, 0.05)
        )

        simulated_weeks = episodes * episode_weeks
        st.caption(
            f"Workload: {simulated_weeks:,} simulated weeks "
            f"({simulated_weeks / 52.0:,.1f} school-year equivalents). "
            "Each episode resets so policies can be compared across reproducible scenarios."
        )
        if simulated_weeks > 50000:
            st.warning("Large runs can take longer. Start with 50-100 episodes before using 1,000.")

    if st.button("Run custom simulation", type="primary", width="stretch"):
        with st.spinner("Running the custom school-support environment..."):
            history, last_info = run_dashboard_simulation(
                episodes=episodes,
                episode_weeks=episode_weeks,
                seed=seed,
                annual_support_budget=annual_support_budget,
                annual_eco_budget=annual_eco_budget,
                start_calendar_week=start_calendar_week,
                demand_multiplier=demand_multiplier,
                weather_intensity=weather_intensity,
                human_error_rate=human_error_rate,
                shock_rate=shock_rate,
            )
        st.session_state["dashboard_history"] = history
        st.session_state["dashboard_last_info"] = last_info

    history = st.session_state.get("dashboard_history")
    last_info = st.session_state.get("dashboard_last_info")
    if history and last_info:
        render_city_metrics(history, last_info)
    else:
        st.info("Run the simulation to see school hub status, helped rates, risks, and municipal outcomes.")


@st.cache_data(show_spinner=False)
def run_dashboard_simulation(
    *,
    episodes: int,
    episode_weeks: int,
    seed: int,
    annual_support_budget: float | None = None,
    annual_eco_budget: float | None = None,
    start_calendar_week: int = 34,
    demand_multiplier: float = 1.0,
    weather_intensity: float = 1.0,
    human_error_rate: float = 1.0,
    shock_rate: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    demand_multiplier = float(np.clip(demand_multiplier, 0.25, 3.0))
    env = CrisisToActionCityEnv(
        seed=seed,
        episode_weeks=episode_weeks,
        school_year_start_week=int(start_calendar_week) % 52,
        district_support_budget=annual_support_budget,
        district_eco_budget=annual_eco_budget,
        weather_intensity=float(np.clip(weather_intensity, 0.0, 3.0)),
        human_error_rate=float(np.clip(human_error_rate, 0.0, 3.0)),
        shock_rate=float(np.clip(shock_rate, 0.0, 3.0)),
        arrival_rate=float(np.clip(DEFAULT_ENV_CONFIG.arrival_rate * demand_multiplier, 0.04, 0.50)),
        rumor_rate=float(np.clip(DEFAULT_ENV_CONFIG.rumor_rate * demand_multiplier, 0.02, 0.40)),
        crisis_rate=float(np.clip(DEFAULT_ENV_CONFIG.crisis_rate * demand_multiplier, 0.01, 0.30)),
    )
    agents = [StreamlitHeuristicAgent(env) for _ in range(env.num_schools)]
    history: list[dict[str, Any]] = []
    last_info: dict[str, Any] = {}

    for episode in range(episodes):
        env.reset(
            seed=seed + episode,
            options={
                "difficulty": demand_multiplier,
                "start_calendar_week": int(start_calendar_week) % 52,
            },
        )
        agent_states = env.get_agent_observations()
        total_reward = 0.0
        helped_sum = 0.0
        wrong_sum = 0.0
        missed_sum = 0.0
        human_error_sum = 0.0
        recovery_sum = 0.0
        fatigue_sum = 0.0
        budget_sum = 0.0
        carbon_sum = 0.0
        carbon_reduction_sum = 0.0
        eco_budget_remaining_sum = 0.0
        eco_budget_spent_sum = 0.0
        unresolved_sum = 0.0
        weather_sum = 0.0
        winter_sum = 0.0
        heat_sum = 0.0
        storm_sum = 0.0
        action_frequency_totals = {name: 0.0 for name in env.ACTION_NAMES}
        initiative_spend_totals = {name: 0.0 for name in getattr(env, "SUSTAINABILITY_ACTION_NAMES", [])}
        initiative_reduction_totals = {name: 0.0 for name in getattr(env, "SUSTAINABILITY_ACTION_NAMES", [])}
        steps = 0

        done = False
        while not done:
            actions = [agent.act(agent_states[idx]) for idx, agent in enumerate(agents)]
            _, reward, terminated, truncated, info = env.step_agents(np.vstack(actions))
            agent_states = env.get_agent_observations()
            total_reward += reward
            helped_sum += float(np.mean([school["people_helped"] for school in info["schools"]]))
            wrong_sum += float(np.mean([school["wrong_guidance"] for school in info["schools"]]))
            missed_sum += float(np.mean([school["missed_urgent"] for school in info["schools"]]))
            human_error_sum += float(np.mean([school.get("human_error", 0.0) for school in info["schools"]]))
            recovery_sum += float(np.mean([school.get("recovery_progress", 0.0) for school in info["schools"]]))
            fatigue_sum += float(np.mean([school.get("fatigue", 0.0) for school in info["schools"]]))
            budget_sum += float(np.mean([school.get("budget_left", 0.0) for school in info["schools"]]))
            carbon_sum += float(info.get("global", {}).get("current_carbon_footprint", 0.0))
            carbon_reduction_sum += float(info.get("global", {}).get("carbon_reduction", 0.0))
            eco_budget_remaining_sum += float(info.get("global", {}).get("eco_budget_remaining", 0.0))
            eco_budget_spent_sum += float(info.get("global", {}).get("eco_budget_spent", 0.0))
            unresolved_sum += float(np.mean([school.get("unresolved_need", 0.0) for school in info["schools"]]))
            weather_sum += float(info.get("global", {}).get("weather_disruption", 0.0))
            winter_sum += float(info.get("global", {}).get("winter_pressure", 0.0))
            heat_sum += float(info.get("global", {}).get("heat_pressure", 0.0))
            storm_sum += float(info.get("global", {}).get("storm_pressure", 0.0))
            for school in info["schools"]:
                for name, value in school.get("last_actions", {}).items():
                    action_frequency_totals[name] = action_frequency_totals.get(name, 0.0) + float(value)
                for name, payload in school.get("sustainability_initiatives", {}).items():
                    initiative_spend_totals[name] = initiative_spend_totals.get(name, 0.0) + float(payload.get("budget_spent", 0.0))
                    initiative_reduction_totals[name] = initiative_reduction_totals.get(name, 0.0) + float(payload.get("co2_reduction", 0.0))
            steps += 1
            done = terminated or truncated
            last_info = info

        history.append(
            {
                "episode": episode,
                "configured_episodes": episodes,
                "configured_episode_weeks": episode_weeks,
                "configured_total_weeks": episodes * episode_weeks,
                "configured_demand_multiplier": demand_multiplier,
                "configured_weather_intensity": weather_intensity,
                "configured_human_error_rate": human_error_rate,
                "configured_shock_rate": shock_rate,
                "reward": total_reward / max(1, steps),
                "episode_total_reward": total_reward,
                "episode_avg_people_helped": helped_sum / max(1, steps),
                "wrong_guidance": wrong_sum / max(1, steps),
                "missed_urgent": missed_sum / max(1, steps),
                "human_error": human_error_sum / max(1, steps),
                "recovery_progress": recovery_sum / max(1, steps),
                "fatigue": fatigue_sum / max(1, steps),
                "budget_left": budget_sum / max(1, steps),
                "current_carbon_footprint": carbon_sum / max(1, steps),
                "initial_carbon_footprint": float(last_info.get("global", {}).get("initial_carbon_footprint", 0.0)),
                "carbon_reduction": carbon_reduction_sum / max(1, steps),
                "eco_budget_remaining": eco_budget_remaining_sum / max(1, steps),
                "eco_budget_spent": eco_budget_spent_sum / max(1, steps),
                "annual_support_budget_total": float(
                    last_info.get("global", {}).get("annual_support_budget_total", 0.0)
                ),
                "annual_support_budget_remaining": float(
                    last_info.get("global", {}).get("annual_support_budget_remaining", 0.0)
                ),
                "annual_support_budget_spent": float(
                    last_info.get("global", {}).get("annual_support_budget_spent", 0.0)
                ),
                "annual_budget_health": float(
                    last_info.get("global", {}).get("annual_budget_health", 0.0)
                ),
                "best_carbon_footprint": float(last_info.get("global", {}).get("best_carbon_footprint", 0.0)),
                "unresolved_need": unresolved_sum / max(1, steps),
                "weather_disruption": weather_sum / max(1, steps),
                "winter_pressure": winter_sum / max(1, steps),
                "heat_pressure": heat_sum / max(1, steps),
                "storm_pressure": storm_sum / max(1, steps),
                "season": str(last_info.get("global", {}).get("season", "unknown")),
                "school_year_week": int(last_info.get("global", {}).get("school_year_week", 0)),
                "school_year": int(last_info.get("global", {}).get("school_year", 0)),
                "daily_weather": list(last_info.get("global", {}).get("daily_weather", [])),
                "action_frequency": action_frequency_totals,
                "initiative_spend": initiative_spend_totals,
                "initiative_reduction": initiative_reduction_totals,
                "hub_statuses": [school["hub_status"] for school in last_info["schools"]],
            }
        )

    return history, last_info


def render_city_metrics(history: list[dict[str, Any]], info: dict[str, Any], *, compact: bool = False) -> None:
    latest = history[-1]
    if compact:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reward / week", f"{latest['reward']:.2f}")
        col2.metric("People helped", f"{latest['episode_avg_people_helped']:.2f}")
        col3.metric("Wrong guidance", f"{latest['wrong_guidance']:.2f}")
        col4.metric("Missed urgent", f"{latest['missed_urgent']:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Human error", f"{latest['human_error']:.2f}")
        col6.metric("Recovery", f"{latest['recovery_progress']:.2f}")
        col7.metric("Fatigue", f"{latest['fatigue']:.2f}")
        col8.metric("Budget left", f"{latest['budget_left']:.2f}")
    else:
        climate = st.columns(4)
        climate[0].metric(
            "Current carbon footprint",
            f"{latest.get('current_carbon_footprint', 0.0):.0f} tCO2/year",
        )
        climate[1].metric(
            "Initial footprint",
            f"{latest.get('initial_carbon_footprint', 0.0):.0f} tCO2/year",
        )
        climate[2].metric("Reduction achieved", f"{latest.get('carbon_reduction', 0.0):.0%}")
        climate[3].metric(
            "Best solution found",
            f"{latest.get('best_carbon_footprint', 0.0):.0f} tCO2/year",
        )

        climate_budget = st.columns(4)
        climate_budget[0].metric(
            "Sustainability budget",
            f"${latest.get('eco_budget_remaining', 0.0) + latest.get('eco_budget_spent', 0.0):,.0f}",
        )
        climate_budget[1].metric(
            "Eco budget remaining",
            f"${latest.get('eco_budget_remaining', 0.0):,.0f}",
        )
        climate_budget[2].metric("Eco budget spent", f"${latest.get('eco_budget_spent', 0.0):,.0f}")
        climate_budget[3].metric("Reward / week", f"{latest['reward']:.2f}")

        st.subheader("Policy Guardrails")
        guardrails = st.columns(4)
        guardrails[0].metric("People helped", f"{latest['episode_avg_people_helped']:.2f}")
        guardrails[1].metric("Wrong guidance", f"{latest['wrong_guidance']:.2f}")
        guardrails[2].metric("Missed urgent", f"{latest['missed_urgent']:.2f}")
        guardrails[3].metric("Fatigue", f"{latest['fatigue']:.2f}")

    if not compact:
        st.subheader("Simulation Coverage")
        coverage = st.columns(4)
        coverage[0].metric("Episodes", f"{latest.get('configured_episodes', len(history)):,}")
        coverage[1].metric("Weeks / episode", f"{latest.get('configured_episode_weeks', 0):,}")
        coverage[2].metric("Total simulated weeks", f"{latest.get('configured_total_weeks', 0):,}")
        coverage[3].metric(
            "School-year equivalents",
            f"{latest.get('configured_total_weeks', 0) / 52.0:,.1f}",
        )

    st.subheader("School-Year Operations")
    budget_cols = st.columns(4)
    budget_cols[0].metric("School year", f"{latest.get('school_year', 0) + 1}")
    budget_cols[1].metric(
        "District annual budget",
        f"${latest.get('annual_support_budget_total', 0.0):,.0f}",
    )
    budget_cols[2].metric(
        "District budget remaining",
        f"${latest.get('annual_support_budget_remaining', 0.0):,.0f}",
    )
    budget_cols[3].metric(
        "District budget spent",
        f"${latest.get('annual_support_budget_spent', 0.0):,.0f}",
    )
    weather_cols = st.columns(4)
    weather_cols[0].metric("Season", str(latest.get("season", "unknown")).title())
    weather_cols[1].metric("School week", f"{latest.get('school_year_week', 0)}")
    weather_cols[2].metric("Weather disruption", f"{latest.get('weather_disruption', 0.0):.2f}")
    weather_cols[3].metric("Winter / Heat", f"{latest.get('winter_pressure', 0.0):.2f} / {latest.get('heat_pressure', 0.0):.2f}")

    if not compact:
        daily_weather = latest.get("daily_weather", [])
        if daily_weather:
            weather_rows = [
                {
                    "Day": day.get("day", ""),
                    "Condition": str(day.get("condition", "")).title(),
                    "Disruption": round(float(day.get("disruption", 0.0)), 2),
                    "Winter": round(float(day.get("winter_pressure", 0.0)), 2),
                    "Heat": round(float(day.get("heat_pressure", 0.0)), 2),
                    "Storm": round(float(day.get("storm_pressure", 0.0)), 2),
                }
                for day in daily_weather
            ]
            st.dataframe(weather_rows, width="stretch", hide_index=True)

        st.subheader("Investment Analysis")
        render_sustainability_analytics(latest, info)

    chart_data = {
        "episode": [row["episode"] for row in history],
        "reward": [row["reward"] for row in history],
        "people_helped": [row["episode_avg_people_helped"] for row in history],
        "recovery": [row["recovery_progress"] for row in history],
        "fatigue": [row["fatigue"] for row in history],
        "carbon_reduction": [row.get("carbon_reduction", 0.0) for row in history],
        "weather_disruption": [row.get("weather_disruption", 0.0) for row in history],
    }
    chart_series = ["reward", "people_helped", "recovery", "fatigue", "weather_disruption"]
    if not compact:
        chart_series.append("carbon_reduction")
    st.line_chart(chart_data, x="episode", y=chart_series)

    if compact:
        render_hub_summary(info)
        return

    st.subheader("School support hubs")
    render_hub_summary(info)

    st.subheader("Latest assistant outputs from the city")
    for idx, school in enumerate(info["schools"]):
        assistant = school.get("assistant_output")
        if not assistant:
            continue
        with st.expander(f"{SCHOOL_NAMES[idx]} - {assistant['case_type']}"):
            st.write(assistant["plain_language_summary"])
            st.write("Next steps:")
            for step in assistant["next_steps"]:
                st.write(f"- {step}")
            st.caption(f"Human review: {assistant['human_review_needed']} | Confidence: {assistant['confidence_label']}")


def render_sustainability_analytics(latest: dict[str, Any], info: dict[str, Any]) -> None:
    spend = latest.get("initiative_spend", {})
    reduction = latest.get("initiative_reduction", {})
    action_frequency = latest.get("action_frequency", {})
    if not spend:
        return

    labels = [name.replace("_", " ").title() for name in spend]
    budget_values = [float(spend[name]) for name in spend]
    action_values = [float(action_frequency.get(name, 0.0)) for name in spend]
    roi_rows = []
    for name in spend:
        cost = float(spend.get(name, 0.0))
        co2 = float(reduction.get(name, 0.0))
        roi_rows.append(
            {
                "Initiative": name.replace("_", " ").title(),
                "Cost": f"${cost:,.0f}",
                "CO2 Reduction": f"{co2:.1f} t",
                "CO2 per $": f"{(co2 / max(1.0, cost)):.5f}",
            }
        )

    col_budget, col_actions = st.columns(2)
    with col_budget:
        st.markdown("**Budget Allocation**")
        st.bar_chart({"Initiative": labels, "Budget": budget_values}, x="Initiative", y="Budget")
    with col_actions:
        st.markdown("**Action Frequency**")
        st.bar_chart({"Initiative": labels, "Times Chosen": action_values}, x="Initiative", y="Times Chosen")

    col_roi, col_twin = st.columns([1.1, 1.0])
    with col_roi:
        st.markdown("**Return on Investment**")
        st.dataframe(roi_rows, width="stretch", hide_index=True)
    with col_twin:
        st.markdown("**School Digital Twin**")
        school = info.get("schools", [{}])[0]
        twin = school.get("digital_twin", {})
        for area, value in twin.items():
            if area == "Renewable Readiness":
                status = "Ready" if value >= 0.60 else "Moderate" if value >= 0.35 else "Low"
            else:
                status = "High" if value >= 0.60 else "Moderate" if value >= 0.35 else "Optimized"
            st.write(f"{area}: {status} ({value:.2f})")


def render_hub_summary(info: dict[str, Any]) -> None:
    cols = st.columns(4)
    for idx, school in enumerate(info["schools"]):
        with cols[idx]:
            status = school["hub_status"]
            st.markdown(
                f"<div style='border:1px solid {STATUS_COLORS.get(status, '#999')};"
                "border-radius:8px;padding:12px;background:#10141c'>"
                f"<h4>{SCHOOL_NAMES[idx]}</h4>"
                f"<b>Status:</b> {status}<br>"
                f"<b>Helped:</b> {school['people_helped']:.2f}<br>"
                f"<b>Unresolved:</b> {school['unresolved_need']:.2f}<br>"
                f"<b>Risk:</b> {max(school['wrong_guidance'], school['missed_urgent']):.2f}<br>"
                f"<b>Fatigue:</b> {school['fatigue']:.2f}<br>"
                f"<b>Weekly budget health:</b> {school['budget_left']:.2f}<br>"
                f"<b>Annual budget left:</b> ${school.get('annual_support_budget_remaining', 0.0):,.0f}<br>"
                f"<b>Weather:</b> {school.get('weather_disruption', 0.0):.2f}<br>"
                f"<b>Transport barrier:</b> {school.get('transportation_barrier', 0.0):.2f}<br>"
                f"<b>Carbon:</b> {school.get('current_carbon_footprint', 0.0):.0f} tCO2/yr<br>"
                f"<b>Carbon reduction:</b> {school.get('carbon_reduction', 0.0):.0%}<br>"
                "</div>",
                unsafe_allow_html=True,
            )


def pitch_dashboard_view() -> None:
    st.header("Pitch Dashboard")

    st.subheader("Devpost Screening Audit")
    for item in DEVPOST_REQUIRED_FIELDS:
        st.write(f"- **{item['field']}**: {item['check']}")
    st.write(f"- **AI Tools Used**: disclose Codex and the technical stack.")

    st.subheader("Problem")
    st.write(
        "Students, parents, and caregivers often miss support because information is scattered, complex, "
        "or hard to trust during stressful moments. Maria is a parent trying not to miss a housing-support deadline."
    )

    st.subheader("Solution")
    st.write(
        "Kitsune turns a confusing request into plain language, a checklist, a confidence signal, "
        "and a human-review flag. The city simulator stress-tests the workflow across school support hubs."
    )

    st.subheader("Why AI")
    st.write(
        "Search gives links. Kitsune combines local Qwen plain-language rewriting with deterministic risk rules "
        "to structure a stressful message, produce action steps, and route sensitive cases to a person."
    )

    st.subheader("AI Architecture")
    st.code(
        "Input: Maria's school support notice\n"
        "  -> deterministic case routing and risk scoring\n"
        "  -> local Qwen plain-language rewrite\n"
        "  -> guarded checklist and source-check prompts\n"
        "  -> deterministic human-review flag\n"
        "  -> RL city simulator for safe staff/budget allocation\n"
        "Output: next steps, confidence, safeguard, and impact metrics",
        language="text",
    )

    st.subheader("Responsible AI Guardrail")
    st.warning(
        "Risk: local Qwen could turn an unverified or maliciously instructed message into confident but wrong guidance.\n\n"
        "Mitigation: generated text never controls decisions. Kitsune sanitizes input, validates output against "
        "deterministic risk rules, falls back to guarded text, and surfaces source checks plus human handoff."
    )

    st.subheader("Human-in-the-loop")
    st.success(
        "Kitsune does not decide whether Maria is eligible for housing support. A school housing liaison or "
        "support staff member checks verified records, current policy, and her context before deciding."
    )

    st.subheader("Data Disclosure")
    st.write(
        "The prototype uses synthetic cases modeled after school notices, food-support forms, counseling pages, "
        "academic-support requests, and source-verification scenarios. No sensitive personal data is used."
    )

if __name__ == "__main__":
    main()
