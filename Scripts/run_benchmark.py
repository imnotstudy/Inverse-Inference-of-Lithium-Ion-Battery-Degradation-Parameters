from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyBaMM benchmark from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML benchmark config file.",
    )
    return parser.parse_args(argv)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


def ensure_within_repo(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if os.path.commonpath([str(resolved_root), str(resolved_path)]) != str(resolved_root):
        raise ValueError(f"Output path must stay inside the repository: {resolved_path}")
    return resolved_path


def load_config(config_arg: str) -> dict[str, Any]:
    root = repo_root()
    config_path = resolve_path(root, config_arg).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_sections = {"inputs", "outputs", "runtime", "solver", "study"}
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ValueError(f"Config is missing top-level sections: {sorted(missing_sections)}")

    outputs = config["outputs"]
    output_root = ensure_within_repo(root, resolve_path(root, outputs["root_dir"]))
    config["_resolved"] = {
        "config_path": config_path,
        "repo_root": root,
        "input_csv": resolve_path(root, config["inputs"]["predictions_csv"]).resolve(),
        "output_root": output_root,
        "checkpoints_dir": ensure_within_repo(root, output_root / outputs["checkpoints_dir"]),
        "summaries_dir": ensure_within_repo(root, output_root / outputs["summaries_dir"]),
        "logs_dir": ensure_within_repo(root, output_root / outputs["logs_dir"]),
    }

    study = config["study"]
    required_columns = study.get("required_columns", [])
    breakpoint_columns = study.get("breakpoint_columns", [])
    stages = study.get("stages", [])
    if not required_columns:
        raise ValueError("study.required_columns must not be empty")
    if not breakpoint_columns:
        raise ValueError("study.breakpoint_columns must not be empty")
    if not stages:
        raise ValueError("study.stages must not be empty")
    if len(stages) != len(breakpoint_columns):
        raise ValueError("study.stages and study.breakpoint_columns must have the same length")

    return config


def configure_logging(logs_dir: Path, level_name: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run_benchmark.log"
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers = [
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path


def validate_predictions(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")


def build_jobs(df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    study = config["study"]
    breakpoint_columns = study["breakpoint_columns"]
    jobs: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        break_points = [int(getattr(row, column)) for column in breakpoint_columns]
        if any(point <= 0 for point in break_points):
            raise ValueError(f"Breakpoints must be positive for id={row.id}: {break_points}")
        if break_points != sorted(break_points):
            raise ValueError(f"Breakpoints must be non-decreasing for id={row.id}: {break_points}")

        jobs.append(
            {
                "id": str(row.id),
                "sei": float(row.sei),
                "plating": float(row.plating),
                "crate": float(row.crate),
                "break_points": break_points,
                "pred_scale": float(row.pred_scale),
            }
        )

    return jobs


def normalize_model_options(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(normalize_model_options(item) for item in value)
    if isinstance(value, dict):
        return {key: normalize_model_options(item) for key, item in value.items()}
    return value


def build_stage_models(stages: list[dict[str, Any]]) -> list[Any]:
    import pybamm

    models = []
    for stage in stages:
        options = normalize_model_options(stage["model_options"])
        models.append(pybamm.lithium_ion.DFN(options))
    return models


def prepare_parameter_values(study: dict[str, Any], job: dict[str, Any]) -> Any:
    import pybamm

    params = pybamm.ParameterValues(study["parameter_values"])
    for name, value in study.get("baseline_parameters", {}).items():
        params[name] = value

    tracked = study["tracked_parameters"]
    params.update(
        {
            tracked["sei"]: job["sei"],
            tracked["plating"]: job["plating"],
        },
        check_already_exists=False,
    )

    scaling = study["concentration_scaling"]
    params[scaling["negative"]["parameter"]] = scaling["negative"]["base_value"] * job["pred_scale"]
    params[scaling["positive"]["parameter"]] = scaling["positive"]["base_value"] * job["pred_scale"]
    return params


def build_experiment(job: dict[str, Any], study: dict[str, Any], cycles: int) -> Any:
    import pybamm

    steps = tuple(step.format(crate=job["crate"]) for step in study["experiment"]["steps"])
    return pybamm.Experiment(
        [steps] * cycles,
        period=study["experiment"]["period"],
        temperature=study["experiment"]["temperature"],
    )


def stage_output_paths(outputs: dict[str, str], job_id: str) -> dict[str, Path]:
    return {
        "checkpoint": Path(outputs["checkpoints_dir"]) / f"{job_id}.pkl",
        "summary": Path(outputs["summaries_dir"]) / f"{job_id}.csv",
    }


def extract_capacity_series(solution: Any, study: dict[str, Any]) -> list[float]:
    summary_key = study.get("summary_variable")
    if summary_key and summary_key in solution.summary_variables:
        return list(solution.summary_variables[summary_key])

    cycle_key = study.get("cycle_capacity_variable", "Discharge capacity [A.h]")
    if solution.cycles:
        return [float(cycle[cycle_key].entries[-1]) for cycle in solution.cycles]

    raise KeyError(
        f"Unable to extract capacity series. Missing summary key '{summary_key}' "
        f"and no cycle data found for '{cycle_key}'."
    )


def run_simulation(job: dict[str, Any], worker_config: dict[str, Any]) -> dict[str, str]:
    import pybamm

    study = worker_config["study"]
    solver_cfg = worker_config["solver"]
    outputs = worker_config["outputs"]

    pybamm.set_logging_level(study["pybamm_log_level"])
    models = build_stage_models(study["stages"])
    params = prepare_parameter_values(study, job)
    paths = stage_output_paths(outputs, job["id"])

    solution = None
    summary_path: Path | None = None

    for index, (stage, model, break_point) in enumerate(zip(study["stages"], models, job["break_points"])):
        previous_break = job["break_points"][index - 1] if index > 0 else 0
        cycles = break_point - previous_break if index > 0 else break_point
        if cycles <= 0:
            raise ValueError(f"Stage '{stage['name']}' has non-positive cycle count for id={job['id']}")

        experiment = build_experiment(job, study, cycles)
        simulation = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=params,
            solver=pybamm.CasadiSolver(
                solver_cfg["mode"],
                dt_max=solver_cfg["dt_max"],
                rtol=solver_cfg["rtol"],
                atol=solver_cfg["atol"],
            ),
        )
        solution = simulation.solve(
            starting_solution=solution,
            save_at_cycles=study["save_at_cycles"],
            calc_esoh=False,
        )

        if stage.get("checkpoint_output"):
            solution.save(paths["checkpoint"])

        if stage.get("summary_output"):
            summary = pd.DataFrame(extract_capacity_series(solution, study), columns=["Capacity"])
            tracked = study["tracked_parameters"]
            summary[tracked["sei"]] = job["sei"]
            summary[tracked["plating"]] = job["plating"]
            summary["Charge rate"] = job["crate"]
            summary["break_point"] = job["break_points"][index - 1] if index > 0 else job["break_points"][0]
            summary_path = paths["summary"]
            summary.to_csv(summary_path, index=False)

    if summary_path is None:
        raise RuntimeError(f"No summary CSV was produced for id={job['id']}")

    return {
        "id": job["id"],
        "checkpoint": str(paths["checkpoint"]),
        "summary": str(summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    resolved = config["_resolved"]

    for path in (resolved["output_root"], resolved["checkpoints_dir"], resolved["summaries_dir"], resolved["logs_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    log_path = configure_logging(resolved["logs_dir"], config["runtime"]["log_level"])
    logging.info("Loaded benchmark config from %s", resolved["config_path"])
    logging.info("Writing benchmark outputs to %s", resolved["output_root"])

    input_csv = resolved["input_csv"]
    if not input_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {input_csv}")

    predictions = pd.read_csv(input_csv)
    validate_predictions(predictions, config["study"]["required_columns"])
    jobs = build_jobs(predictions, config)
    if not jobs:
        raise ValueError("Predictions CSV does not contain any benchmark rows")

    worker_config = {
        "study": config["study"],
        "solver": config["solver"],
        "outputs": {
            "checkpoints_dir": str(resolved["checkpoints_dir"]),
            "summaries_dir": str(resolved["summaries_dir"]),
        },
    }

    max_workers = int(config["runtime"]["max_workers"])
    logging.info("Starting %s benchmark job(s) with max_workers=%s", len(jobs), max_workers)

    def log_completed(result: dict[str, str]) -> None:
        logging.info(
            "Completed benchmark job for id=%s | checkpoint=%s | summary=%s",
            result["id"],
            result["checkpoint"],
            result["summary"],
        )

    if max_workers == 1:
        for job in jobs:
            try:
                result = run_simulation(job, worker_config)
            except Exception:
                logging.exception("Benchmark job failed for id=%s", job["id"])
                raise
            log_completed(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(run_simulation, job, worker_config): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception:
                    logging.exception("Benchmark job failed for id=%s", job["id"])
                    raise
                log_completed(result)

    logging.info("Benchmark run finished successfully. Log file: %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
