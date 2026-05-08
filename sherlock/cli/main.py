"""Top-level `sherlock` CLI."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from sherlock.agent import Sherlock
from sherlock.config import Config

app = typer.Typer(
    name="sherlock",
    help="Domain-agnostic context-curation library — main LLM authors its companions, "
    "memory fades naturally, intent is inferred.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
config_app = typer.Typer(name="config", help="Inspect/validate the YAML config.", no_args_is_help=True)
app.add_typer(config_app)


def _resolve_config(path: Path | None) -> Path:
    if path is not None:
        return path
    candidate = Path("sherlock.yaml")
    if candidate.exists():
        return candidate
    raise typer.BadParameter(
        "No --config provided and ./sherlock.yaml not found. "
        "Pass --config <path> or run from a directory with sherlock.yaml.",
        param_hint="--config",
    )


@app.command()
def chat(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to sherlock.yaml"),
    one_shot: str | None = typer.Option(
        None,
        "--one-shot",
        help="Send a single user message, print the reply, and exit. "
        "If omitted, runs an interactive REPL.",
    ),
) -> None:
    """Start an interactive chat (or one-shot when --one-shot is given)."""
    cfg_path = _resolve_config(config)
    cfg = Config.from_yaml(cfg_path)
    agent = Sherlock(cfg)

    if one_shot is not None:
        reply = agent.chat(one_shot)
        console.print(reply)
        return

    console.print(
        Panel.fit(
            f"[bold]Sherlock[/bold] — {cfg.project}\n"
            f"main: {cfg.models.main.provider}/{cfg.models.main.model}\n"
            "Type your message. /exit or Ctrl-D to quit.",
            border_style="cyan",
        )
    )
    while True:
        try:
            user = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            console.print("[dim]bye[/dim]")
            return
        try:
            reply = agent.chat(user)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]error:[/red] {exc}", style="red")
            continue
        console.print(Markdown(reply))


@config_app.command("validate")
def config_validate(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Validate that the YAML parses, references resolve, and the prompt file exists."""
    cfg_path = _resolve_config(config)
    try:
        cfg = Config.from_yaml(cfg_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]invalid:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]ok[/green]  project={cfg.project}")
    console.print(f"  main_system_prompt: {cfg.main_system_prompt.path}")
    console.print(f"  main model: {cfg.models.main.provider}/{cfg.models.main.model}")
    console.print(f"  storage.sqlite: {cfg.storage.sqlite_path}")


@config_app.command("show")
def config_show(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Print the parsed config (sensitive values redacted)."""
    cfg_path = _resolve_config(config)
    cfg = Config.from_yaml(cfg_path)
    console.print_json(data=cfg.model_dump(mode="json"))


@app.command("models")
def models_list(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """List the providers/models the current config references."""
    cfg_path = _resolve_config(config)
    cfg = Config.from_yaml(cfg_path)
    console.print(f"main: {cfg.models.main.provider}/{cfg.models.main.model}")
    if cfg.models.background_summary:
        console.print(
            f"summary: {cfg.models.background_summary.provider}/{cfg.models.background_summary.model}"
        )
    if cfg.models.background_inference:
        console.print(
            f"inference: {cfg.models.background_inference.provider}/{cfg.models.background_inference.model}"
        )


@app.command("evaluate")
def evaluate(
    config: Path | None = typer.Option(None, "--config", "-c"),
    conversation: Path = typer.Option(
        Path("evaluation/dummy_conversation.md"),
        "--conversation",
        help="Path to dummy conversation markdown.",
    ),
    gold: Path = typer.Option(
        Path("evaluation/gold_standard.md"),
        "--gold",
        help="Path to gold standard markdown.",
    ),
    evaluator_prompt: Path = typer.Option(
        Path("evaluation/evaluator_system_prompt.txt"),
        "--evaluator-prompt",
        help="Evaluator rubric file.",
    ),
    runs_root: Path = typer.Option(
        Path("evaluation/runs"),
        "--runs",
        help="Directory under which the timestamped run is written.",
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", help="Cap turns for fast smoke runs."),
    skip_score: bool = typer.Option(False, "--skip-score", help="Replay + format only; don't call the evaluator."),
) -> None:
    """Replay the dummy conversation through Sherlock and score against the gold standard."""
    from sherlock.evaluation import format_sherlock_output, replay_dummy_conversation
    from sherlock.evaluation.evaluator import GeminiEvaluator

    cfg_path = _resolve_config(config)
    cfg = Config.from_yaml(cfg_path)
    # from_yaml installs companion prompts (bootstrapped or default) and
    # wires the search engine, so the memory layer is fully active.
    agent = Sherlock.from_yaml(cfg_path)

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = runs_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Replaying[/cyan] {conversation} → {cfg.models.main.provider}/{cfg.models.main.model}")

    import time as _time

    _start = _time.monotonic()

    def _progress(i, t, error):
        elapsed = _time.monotonic() - _start
        if error:
            console.print(f"  turn {t.turn_number} [red]ERR[/red] {error}")
        else:
            if (i + 1) % 5 == 0:
                rate = (i + 1) / elapsed if elapsed else 0
                eta_min = (80 - (i + 1)) / rate / 60 if rate else 0
                console.print(
                    f"  replayed {i + 1} turns ({elapsed:.0f}s elapsed, "
                    f"{rate:.2f} turns/s, ETA ~{eta_min:.1f} min for 80 turns)"
                )

    turns = replay_dummy_conversation(
        agent, conversation, max_turns=max_turns, progress_callback=_progress
    )
    console.print(f"[green]Replay done[/green] — {len(turns)} turns")

    output = format_sherlock_output(agent)
    candidate_path = run_dir / "sherlock_output.md"
    candidate_path.write_text(output.to_markdown(), encoding="utf-8")
    console.print(f"  wrote {candidate_path}")

    if skip_score:
        console.print("[yellow]--skip-score set; not calling evaluator.[/yellow]")
        return

    evaluator = GeminiEvaluator(evaluator_prompt)
    gold_md = gold.read_text(encoding="utf-8")
    score = evaluator.evaluate(gold_md=gold_md, candidate_md=output.to_markdown())

    eval_json = run_dir / "evaluator_output.json"
    eval_json.write_text(
        json.dumps({**score.to_dict(), "raw_response": score.raw_response}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "score.txt").write_text(str(score.final_score) + "\n", encoding="utf-8")
    (run_dir / "comparison_input.md").write_text(
        f"# GOLD\n\n{gold_md}\n\n---\n\n# CANDIDATE\n\n{output.to_markdown()}",
        encoding="utf-8",
    )

    console.print(
        Panel.fit(
            f"[bold]Final: {score.final_score}/100[/bold]\n"
            f"  summary_fidelity:           {score.summary_fidelity}\n"
            f"  inference_quality:          {score.inference_quality}\n"
            f"  classification_correctness: {score.classification_correctness}\n"
            f"  tool_recommendations:       {score.tool_recommendations}\n\n"
            f"notes: {score.notes[:300]}{'…' if len(score.notes) > 300 else ''}",
            border_style="cyan" if score.final_score >= 80 else "yellow",
        )
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
