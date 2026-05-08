"""Top-level `sherlock` CLI. M1 surface only — `chat` and `config`."""
from __future__ import annotations

import sys
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


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
