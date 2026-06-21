from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from school_help_env import (
    ACTION_LABELS,
    SCHOOL_NAMES,
    CrisisToActionCityEnv,
    MunicipalLeague,
    _accumulate_episode_metrics,
    _empty_episode_metric_sums,
    _finalize_episode_metrics,
)


WIDTH = 1380
HEIGHT = 900
FPS = 60
CARD_RADIUS = 10


COLORS = {
    "bg": (18, 21, 27),
    "panel": (31, 36, 46),
    "panel_2": (39, 45, 57),
    "line": (65, 74, 92),
    "text": (232, 237, 245),
    "muted": (157, 168, 184),
    "green": (70, 210, 132),
    "yellow": (245, 190, 78),
    "red": (238, 95, 95),
    "blue": (100, 166, 255),
    "magenta": (210, 120, 255),
    "cyan": (82, 210, 220),
    "orange": (245, 142, 80),
}


STATUS_COLORS = {
    "active": COLORS["blue"],
    "probation": COLORS["yellow"],
    "recovery": COLORS["cyan"],
    "model_hub": COLORS["green"],
    "collapsed": COLORS["red"],
}


@dataclass
class ViewerState:
    episode: int = 0
    week: int = 0
    paused: bool = False
    speed: float = 1.0
    total_reward: float = 0.0
    last_reward: float = 0.0
    last_info: dict[str, Any] | None = None
    last_report: dict[str, Any] | None = None
    event_log: list[str] | None = None

    def __post_init__(self) -> None:
        if self.event_log is None:
            self.event_log = []


class HeuristicAgent:
    def __init__(self, school_id: int, env: CrisisToActionCityEnv) -> None:
        self.school_id = school_id
        self.local_names = list(env.LOCAL_OBS_NAMES)
        self.action_bias = float(env.config.action_activation_bias)

    def act(self, state: np.ndarray) -> tuple[np.ndarray, None, None]:
        local = {name: float(state[idx]) for idx, name in enumerate(self.local_names)}
        confusing = local["confusing_documents"]
        urgent = local["urgent_cases"]
        rumors = local["rumor_load"]
        unmatched = local["unmatched_requests"]
        followup = local["followup_backlog"]
        trust = local["trust"]
        clarity = local["clarity"]
        sources = local["source_confidence"]
        fatigue = local["worker_fatigue"]
        language = local["language_barrier"]
        service = local["service_availability"]
        misinformation = local["misinformation_pressure"]

        desired = np.array(
            [
                0.30 + 0.58 * confusing + 0.22 * language + 0.18 * (1.0 - clarity) + 0.10 * local.get("weather_disruption", 0.0),
                0.24 + 0.68 * rumors + 0.26 * misinformation + 0.16 * (1.0 - sources),
                0.36 + 0.70 * unmatched + 0.18 * service + 0.08 * trust,
                0.32 + 0.78 * urgent + 0.22 * local["last_missed_urgent"] + 0.20 * local.get("weather_disruption", 0.0),
                0.34 + 0.76 * followup + 0.26 * unmatched + 0.20 * local.get("transportation_barrier", 0.0),
                0.18 + 0.40 * fatigue + 0.18 * (1.0 - local["worker_capacity"]) + 0.12 * local.get("attendance_disruption", 0.0),
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
        return (logits + self.action_bias).astype(np.float32), None, None


class CityViewer:
    def __init__(
        self,
        agents: list[Any] | None = None,
        *,
        seed: int | None = 42,
        report_every: int = 25,
        env_settings: dict[str, Any] | None = None,
        school_settings: list[dict[str, float]] | None = None,
        headless: bool = False,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("Kitsune City Simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small = pygame.font.SysFont("segoeui", 15)
        self.tiny = pygame.font.SysFont("segoeui", 13)
        self.big = pygame.font.SysFont("segoeui", 28, bold=True)
        self.medium = pygame.font.SysFont("segoeui", 20, bold=True)

        self.seed = seed
        self.env_settings = dict(env_settings or {})
        self.school_settings = school_settings
        self.env = CrisisToActionCityEnv(seed=seed, school_settings=school_settings, **self.env_settings)
        self.agents = agents or [HeuristicAgent(idx, self.env) for idx in range(4)]
        self.league = MunicipalLeague(
            self.env.school_settings,
            report_every=report_every,
            winner_grant=0.06,
            runner_up_grant=0.025,
            fine=0.025,
            training_bonus=0.22,
            training_fine=0.08,
            min_budget=0.45,
            survival_league=True,
            color=False,
            seed=seed,
        )
        self.state = ViewerState()
        self.agent_states: list[np.ndarray] = []
        self.episode_scores = np.zeros(4, dtype=np.float32)
        self.metric_sums = _empty_episode_metric_sums(4)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.league.apply_to_env(self.env)
        _, info = self.env.reset(seed=None if self.seed is None else self.seed + self.state.episode)
        self.agent_states = self.env.get_agent_observations()
        self.episode_scores = np.zeros(self.env.num_schools, dtype=np.float32)
        self.metric_sums = _empty_episode_metric_sums(self.env.num_schools)
        self.state.week = 0
        self.state.total_reward = 0.0
        self.state.last_reward = 0.0
        self.state.last_info = info

    def run(self, *, max_frames: int | None = None) -> None:
        running = True
        frame = 0
        accumulator = 0.0
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(event.key)

            if not self.state.paused:
                accumulator += dt * self.state.speed
                while accumulator >= 0.25:
                    self.step()
                    accumulator -= 0.25

            self.draw()
            pygame.display.flip()
            if max_frames is not None and frame >= max_frames:
                running = False

        pygame.quit()

    def handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.state.paused = not self.state.paused
        elif key == pygame.K_n:
            self.step()
        elif key == pygame.K_r:
            self.state.episode += 1
            self.reset_episode()
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_UP):
            self.state.speed = min(8.0, self.state.speed * 1.35)
        elif key in (pygame.K_MINUS, pygame.K_DOWN):
            self.state.speed = max(0.25, self.state.speed / 1.35)
        return True

    def step(self) -> None:
        actions = []
        for idx, agent in enumerate(self.agents):
            if hasattr(agent, "act"):
                action, _, _ = agent.act(self.agent_states[idx])
            else:
                action = agent(self.agent_states[idx])
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_array.size == getattr(self.env, "SUPPORT_ACTION_COUNT", 6):
                padded = np.zeros(self.env.agent_action_dim, dtype=np.float32)
                padded[: action_array.size] = action_array
                action_array = padded
            actions.append(action_array.reshape(self.env.agent_action_dim))

        _, reward, terminated, truncated, info = self.env.step_agents(np.vstack(actions))
        self.agent_states = self.env.get_agent_observations()
        self.state.last_reward = float(reward)
        self.state.total_reward += float(reward)
        self.state.last_info = info
        self.state.week = int(info["week"])

        for idx, school in enumerate(info["schools"]):
            self.episode_scores[idx] += float(school["local_reward"])
        _accumulate_episode_metrics(self.metric_sums, info)

        if terminated or truncated:
            episode_metrics = _finalize_episode_metrics(
                self.episode_scores,
                self.metric_sums,
                max(1, self.state.week),
                info,
            )
            report = self.league.record_episode(self.state.episode, self.episode_scores, info, episode_metrics)
            if report is not None:
                self.league.apply_to_env(self.env)
                self.state.last_report = report
                self.push_log(self.short_report(report))
            reward_per_week = self.state.total_reward / max(1, self.state.week)
            self.push_log(
                f"Episode {self.state.episode} done | reward/wk={reward_per_week:.2f} "
                f"| total={self.state.total_reward:.2f} "
                f"| helped_ep={episode_metrics['episode_avg_people_helped']:.2f}"
            )
            self.state.episode += 1
            self.reset_episode()

    def push_log(self, text: str) -> None:
        self.state.event_log.append(text)
        self.state.event_log = self.state.event_log[-7:]

    def short_report(self, report: dict[str, Any]) -> str:
        winner = report.get("winner")
        collapsed = report.get("collapsed_hubs", [])
        model = report.get("model_hub")
        return f"Municipal report #{report['report_number']} | winner=Agent {winner + 1 if winner is not None else '?'} | model={model} | collapsed={collapsed}"

    def draw(self) -> None:
        self.screen.fill(COLORS["bg"])
        info = self.state.last_info or {"schools": [], "global": {}}
        self.draw_header(info)
        self.draw_school_cards(info)
        self.draw_footer(info)

    def draw_header(self, info: dict[str, Any]) -> None:
        global_info = info.get("global", {})
        title = "Kitsune City Simulator"
        self.draw_text(title, self.big, COLORS["text"], 28, 18)
        subtitle = (
            f"Episode {self.state.episode} | Week {self.state.week:02d}/{self.env.config.episode_weeks} "
            f"| reward/wk={self.state.total_reward / max(1, self.state.week):.2f} "
            f"| speed={self.state.speed:.2f}x "
            f"| shared sources={global_info.get('shared_source_library', 0.0):.2f} "
            f"| staff pool={global_info.get('district_staff_pool', 0.0):.2f} "
            f"| shock={global_info.get('last_system_shock', 0.0):.2f}"
        )
        self.draw_text(subtitle, self.font, COLORS["muted"], 30, 56)
        controls = "SPACE pause | N step | R reset episode | UP/DOWN speed | ESC quit"
        self.draw_text(controls, self.small, COLORS["muted"], 30, 82)

    def draw_school_cards(self, info: dict[str, Any]) -> None:
        cards = [(28, 118), (704, 118), (28, 470), (704, 470)]
        for idx, school in enumerate(info.get("schools", [])):
            self.draw_school_card(idx, school, cards[idx][0], cards[idx][1], 648, 324)

    def draw_school_card(self, idx: int, school: dict[str, Any], x: int, y: int, w: int, h: int) -> None:
        status = school.get("hub_status", "active")
        border = STATUS_COLORS.get(status, COLORS["line"])
        pygame.draw.rect(self.screen, COLORS["panel"], (x, y, w, h), border_radius=CARD_RADIUS)
        pygame.draw.rect(self.screen, border, (x, y, w, h), width=2, border_radius=CARD_RADIUS)

        name = SCHOOL_NAMES[idx % len(SCHOOL_NAMES)]
        self.draw_text(f"Agent {idx + 1} / {name}", self.medium, COLORS["text"], x + 18, y + 14)
        self.draw_pill(status, x + w - 150, y + 16, 126, 28, border)
        self.draw_text(
            f"reward={school['local_reward']:+.2f}  helped={school['people_helped']:.2f}  unresolved={school['unresolved_need']:.2f}",
            self.small,
            COLORS["muted"],
            x + 18,
            y + 48,
        )

        metric_x = x + 18
        metric_y = y + 76
        self.draw_bar("trust", school["trust"], metric_x, metric_y, 185)
        self.draw_bar("clarity", school["clarity"], metric_x + 205, metric_y, 185)
        self.draw_bar("sources", school["source_confidence"], metric_x + 410, metric_y, 185)
        self.draw_bar("budget", school["budget_left"], metric_x, metric_y + 34, 185)
        self.draw_bar("fatigue", school["fatigue"], metric_x + 205, metric_y + 34, 185, invert=True)
        self.draw_bar(
            "risk",
            max(school["wrong_guidance"], school["missed_urgent"], school.get("human_error", 0.0)),
            metric_x + 410,
            metric_y + 34,
            185,
            invert=True,
        )

        actions = school.get("last_actions", {})
        ax = x + 18
        ay = y + 154
        for action_idx, (key, label) in enumerate(ACTION_LABELS.items()):
            row = action_idx % 3
            col = action_idx // 3
            self.draw_action_bar(label, float(actions.get(key, 0.0)), ax + col * 305, ay + row * 30, 270)

        assistant = school.get("assistant_output") or {}
        case_line = f"{assistant.get('user', 'Synthetic user')} -> {assistant.get('case_type', 'case')}"
        self.draw_text(case_line, self.small, COLORS["cyan"], x + 18, y + 255)
        summary = assistant.get("plain_language_summary", "Waiting for next case.")
        self.draw_wrapped(summary, self.tiny, COLORS["text"], x + 18, y + 278, w - 36, max_lines=2)
        review = "YES" if assistant.get("human_review_needed") else "no"
        confidence = assistant.get("confidence_label", "n/a")
        self.draw_text(f"confidence={confidence} | human_review={review}", self.tiny, COLORS["muted"], x + 18, y + 306)

    def draw_footer(self, info: dict[str, Any]) -> None:
        y = 810
        pygame.draw.rect(self.screen, COLORS["panel_2"], (28, y, WIDTH - 56, 68), border_radius=8)
        self.draw_text("Event log", self.small, COLORS["muted"], 44, y + 8)
        for idx, line in enumerate(self.state.event_log[-3:]):
            self.draw_text(line, self.tiny, COLORS["text"], 44, y + 29 + idx * 16)

        statuses = [school.get("hub_status", "active") for school in info.get("schools", [])]
        collapsed = [idx + 1 for idx, status in enumerate(statuses) if status == "collapsed"]
        model = next((idx + 1 for idx, status in enumerate(statuses) if status == "model_hub"), None)
        self.draw_text(f"model_hub={model} | collapsed={collapsed}", self.small, COLORS["muted"], WIDTH - 320, y + 14)

    def draw_bar(self, label: str, value: float, x: int, y: int, w: int, *, invert: bool = False) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        color = self.value_color(value, invert=invert)
        self.draw_text(f"{label} {value:.2f}", self.tiny, COLORS["muted"], x, y - 14)
        pygame.draw.rect(self.screen, (21, 24, 31), (x, y, w, 12), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, int(w * value), 12), border_radius=4)

    def draw_action_bar(self, label: str, value: float, x: int, y: int, w: int) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        self.draw_text(f"{label[:18]:18s}", self.tiny, COLORS["muted"], x, y)
        pygame.draw.rect(self.screen, (21, 24, 31), (x + 112, y + 4, w - 118, 9), border_radius=4)
        pygame.draw.rect(self.screen, COLORS["blue"], (x + 112, y + 4, int((w - 118) * value), 9), border_radius=4)
        self.draw_text(f"{value:.2f}", self.tiny, COLORS["text"], x + w - 38, y)

    def draw_pill(self, text: str, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=14)
        surf = self.small.render(text, True, (10, 13, 18))
        self.screen.blit(surf, (x + (w - surf.get_width()) // 2, y + (h - surf.get_height()) // 2))

    def draw_text(self, text: str, font: pygame.font.Font, color: tuple[int, int, int], x: int, y: int) -> None:
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))

    def draw_wrapped(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        width: int,
        *,
        max_lines: int,
    ) -> None:
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        for idx, line in enumerate(lines[:max_lines]):
            self.draw_text(line, font, color, x, y + idx * 16)

    def value_color(self, value: float, *, invert: bool = False) -> tuple[int, int, int]:
        good = value <= 0.35 if invert else value >= 0.70
        mid = value <= 0.65 if invert else value >= 0.40
        if good:
            return COLORS["green"]
        if mid:
            return COLORS["yellow"]
        return COLORS["red"]


def run_pygame_city_viewer(
    agent1: Any | None = None,
    agent2: Any | None = None,
    agent3: Any | None = None,
    agent4: Any | None = None,
    *,
    seed: int | None = 42,
    report_every: int = 25,
    env_settings: dict[str, Any] | None = None,
    school_settings: list[dict[str, float]] | None = None,
) -> None:
    agents = [agent for agent in (agent1, agent2, agent3, agent4) if agent is not None]
    if len(agents) not in (0, 4):
        raise ValueError("Pass either no agents for demo mode or exactly four PPO agents.")
    viewer = CityViewer(
        agents=agents or None,
        seed=seed,
        report_every=report_every,
        env_settings=env_settings,
        school_settings=school_settings,
    )
    viewer.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Kitsune City Simulator with Pygame.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--headless-test", action="store_true")
    args = parser.parse_args()

    viewer = CityViewer(seed=args.seed, report_every=args.report_every, headless=args.headless_test)
    viewer.run(max_frames=5 if args.headless_test else None)


if __name__ == "__main__":
    main()
