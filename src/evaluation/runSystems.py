from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.evaluation.DataLoader import DataLoader
from src.evaluation.EvaluationHarness import EvaluationHarness
from src.evaluation.runners import (
    MediumAgentSystemRunner,
    MultiAgentSystemRunner,
    SingleAgentSystemRunner,
)
from src.settings import settings


SYSTEM_IDS = ("single_agent_system", "medium_agent_system", "multi_agent_system")


def build_harness() -> EvaluationHarness:
    data_loader = DataLoader(str(Path(settings.DATASET_DIRECTORY).expanduser()))
    return EvaluationHarness(
        data_loader=data_loader,
        output_dir=Path(settings.EVALUATION_DIRECTORY).expanduser(),
    )


def register_systems(harness: EvaluationHarness, selected_system: str = "all") -> None:
    selected = set(SYSTEM_IDS if selected_system == "all" else (selected_system,))

    if "single_agent_system" in selected:
        harness._register_system(
            SingleAgentSystemRunner(description="Single agent baseline")
        )

    if "medium_agent_system" in selected:
        harness._register_system(
            MediumAgentSystemRunner(description="Medium agent system with supervisor and executor")
        )

    if "multi_agent_system" in selected:
        harness._register_system(
            MultiAgentSystemRunner(description="Full multi-agent SDLC system")
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered agent systems for evaluation.")
    parser.add_argument(
        "--datapoint",
        default="snake_game",
        help="Dataset datapoint id to run. Ignored when --all-datapoints is set.",
    )
    parser.add_argument(
        "--all-datapoints",
        action="store_true",
        help="Run every datapoint from the configured dataset.",
    )
    parser.add_argument(
        "--system",
        choices=("all", *SYSTEM_IDS),
        default="all",
        help="Which system to run.",
    )
    parser.add_argument(
        "--human-approval",
        action="store_true",
        help="Enable human approval prompts where a runner supports them.",
    )
    parser.add_argument(
        "--debug-structured-output",
        action="store_true",
        help="Enable structured-output debugging in runners that support it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print compact graph stage and command progress.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    harness = build_harness()
    register_systems(harness, args.system)

    if args.all_datapoints:
        if args.system == "all":
            harness.run_all_datapoints_with_all_systems(
                human_approval=args.human_approval,
                debug_structured_output=args.debug_structured_output,
                verbose=args.verbose,
            )
        else:
            harness.run_all_datapoints_with_system(
                system_id=args.system,
                human_approval=args.human_approval,
                debug_structured_output=args.debug_structured_output,
                verbose=args.verbose,
            )
        return 0

    if args.system == "all":
        harness.run_datapoint_with_all_systems(
            datapoint_id=args.datapoint,
            human_approval=args.human_approval,
            debug_structured_output=args.debug_structured_output,
            verbose=args.verbose,
        )
    else:
        harness.run_datapoint_with_system(
            datapoint_id=args.datapoint,
            human_approval=args.human_approval,
            system_id=args.system,
            debug_structured_output=args.debug_structured_output,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
