"""CLI entry point for Teams Attendant."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="teams-attendant",
    help="AI agent that attends Microsoft Teams meetings on your behalf.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def join(
    meeting_url: str = typer.Argument(..., help="Teams meeting URL to join"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Behavior profile"),
    vision: bool = typer.Option(False, "--vision", "-v", help="Enable vision mode"),
    user_name: str = typer.Option("", "--name", "-n", help="Your display name (for address detection)"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level"),
    browser: str = typer.Option("", "--browser", "-b", help="Browser: chromium (default) or msedge"),
) -> None:
    """Join a Teams meeting with the AI agent."""
    from teams_attendant.config import load_app_config
    from teams_attendant.orchestrator import MeetingOrchestrator
    from teams_attendant.utils.logging import setup_logging

    setup_logging(level=log_level)

    config = load_app_config()

    if browser:
        from teams_attendant.config import merge_configs
        config = merge_configs(config, {"browser": browser})

    if "teams.microsoft.com" not in meeting_url and "teams.live.com" not in meeting_url:
        console.print("[red]Error:[/red] Invalid Teams meeting URL.")
        raise typer.Exit(code=1)

    console.print("[bold]Teams Attendant[/bold]")
    console.print(f"  Profile: [cyan]{profile}[/cyan]")
    console.print(f"  Vision:  [cyan]{'enabled' if vision else 'disabled'}[/cyan]")
    console.print(f"  Browser: [cyan]{config.browser}[/cyan]")
    console.print()

    orchestrator = MeetingOrchestrator(config)
    try:
        asyncio.run(
            orchestrator.join_meeting(
                meeting_url=meeting_url,
                profile_name=profile,
                vision_enabled=vision,
                user_name=user_name,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        asyncio.run(orchestrator.leave_meeting())
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def login(
    clear: bool = typer.Option(False, "--clear", help="Clear existing session and re-login"),
    browser: str = typer.Option("", "--browser", "-b", help="Browser: chromium (default) or msedge"),
) -> None:
    """Authenticate with Microsoft Teams (first-time setup)."""
    from teams_attendant.browser.auth import clear_session, is_session_valid, login as teams_login
    from teams_attendant.config import load_app_config

    config = load_app_config()
    browser_type = browser or config.browser

    if clear:
        console.print("Clearing existing session...")
        asyncio.run(clear_session(config.browser_data_dir))
        console.print("[green]✓[/green] Session cleared.")

    console.print("Checking existing session...")
    if asyncio.run(is_session_valid(config.browser_data_dir, browser=browser_type)):
        console.print("[green]✓[/green] Already logged in!")
        return

    console.print("Opening browser for Teams login...")
    console.print("[dim]Please sign in to your Microsoft Teams account.[/dim]")
    console.print("[dim]The browser will close automatically once login is detected.[/dim]")
    console.print()

    try:
        asyncio.run(teams_login(config.browser_data_dir, browser=browser_type))
        console.print("[green]✓[/green] Login successful! Session saved.")
    except Exception as e:
        console.print(f"[red]✗[/red] Login failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def config(
    show: bool = typer.Option(True, "--show/--no-show", help="Display current configuration"),
) -> None:
    """View agent configuration."""
    from teams_attendant.config import get_config_dir, load_app_config

    app_config = load_app_config()
    config_dir = get_config_dir()

    console.print("[bold]Configuration[/bold]")
    console.print(f"  Config dir:      {config_dir}")
    console.print(f"  Browser data:    {app_config.browser_data_dir}")
    console.print(f"  Summaries dir:   {app_config.summaries_dir}")
    console.print(f"  Default profile: {app_config.default_profile}")
    console.print(f"  Browser:         {app_config.browser}")
    console.print()
    console.print("[bold]Azure Speech[/bold]")
    console.print(f"  Region: {app_config.azure.speech.region}")
    console.print(f"  Key:    {'***configured***' if app_config.azure.speech.key else '[red]not set[/red]'}")
    console.print()
    console.print("[bold]Azure Foundry[/bold]")
    console.print(f"  Endpoint:   {app_config.azure.foundry.endpoint or '[red]not set[/red]'}")
    console.print(f"  Model:      {app_config.azure.foundry.model_deployment}")
    console.print(f"  API Key:    {'***configured***' if app_config.azure.foundry.api_key else '[red]not set[/red]'}")


@app.command()
def profiles(
    action: str = typer.Argument("list", help="Action: list or show"),
    name: str = typer.Argument(None, help="Profile name (for 'show' action)"),
) -> None:
    """Manage behavior profiles."""
    from teams_attendant.config import list_profiles as get_profiles
    from teams_attendant.config import load_profile

    if action == "list":
        all_profiles = get_profiles()
        if not all_profiles:
            console.print("[yellow]No profiles found.[/yellow]")
            return

        table = Table(title="Behavior Profiles")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Proactivity", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Response Length")

        for p in all_profiles:
            table.add_row(
                p.name,
                p.description,
                f"{p.proactivity:.1f}",
                f"{p.response_threshold:.1f}",
                p.response_length,
            )
        console.print(table)

    elif action == "show":
        if not name:
            console.print("[red]Error:[/red] Profile name required for 'show' action.")
            raise typer.Exit(code=1)
        try:
            p = load_profile(name)
        except (FileNotFoundError, Exception) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

        console.print(f"[bold]Profile: {p.name}[/bold]")
        console.print(f"  Description:       {p.description}")
        console.print(f"  Response threshold: {p.response_threshold}")
        console.print(f"  Proactivity:       {p.proactivity}")
        console.print(f"  Response length:   {p.response_length}")
        console.print(f"  Prefer voice:      {p.prefer_voice}")
        console.print(f"  Cooldown:          {p.cooldown_seconds}s")
    else:
        console.print(f"[red]Unknown action:[/red] {action}. Use 'list' or 'show'.")
        raise typer.Exit(code=1)


@app.command()
def summaries(
    action: str = typer.Argument("list", help="Action: list or show"),
    meeting_id: str = typer.Argument(None, help="Meeting summary ID (for 'show' action)"),
) -> None:
    """View meeting summaries."""
    from teams_attendant.agent.summarizer import MeetingSummarizer
    from teams_attendant.config import load_app_config

    app_config = load_app_config()
    summarizer = MeetingSummarizer(llm_client=None, summaries_dir=app_config.summaries_dir)

    if action == "list":
        all_summaries = summarizer.list_summaries()
        if not all_summaries:
            console.print("[yellow]No meeting summaries found.[/yellow]")
            return

        table = Table(title="Meeting Summaries")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Date")

        for s in all_summaries:
            table.add_row(s.get("id", ""), s.get("title", ""), s.get("date", ""))
        console.print(table)

    elif action == "show":
        if not meeting_id:
            console.print("[red]Error:[/red] Meeting ID required for 'show' action.")
            raise typer.Exit(code=1)

        content = summarizer.load_summary(meeting_id)
        if content is None:
            console.print(f"[red]Summary not found:[/red] {meeting_id}")
            raise typer.Exit(code=1)

        from rich.markdown import Markdown

        console.print(Markdown(content))
    else:
        console.print(f"[red]Unknown action:[/red] {action}. Use 'list' or 'show'.")
        raise typer.Exit(code=1)


@app.command(name="audio-check")
def audio_check() -> None:
    """Check virtual audio device setup."""
    from teams_attendant.audio.devices import check_audio_setup

    status = check_audio_setup()
    console.print("[bold]Audio Setup Check[/bold]")
    console.print(f"  Platform: {status.platform}")
    if status.capture_device:
        console.print(f"  [green]✓[/green] Capture device: {status.capture_device}")
    if status.playback_device:
        console.print(f"  [green]✓[/green] Playback device: {status.playback_device}")
    for issue in status.issues:
        console.print(f"  [red]✗[/red] {issue}")
    for suggestion in status.suggestions:
        console.print(f"    → {suggestion}")
    if status.is_ready:
        console.print("\n[green]✓ Audio setup is ready![/green]")
    else:
        console.print("\n[red]✗ Audio setup needs attention.[/red]")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show version information."""
    from teams_attendant import __version__

    console.print(f"Teams Attendant v{__version__}")


if __name__ == "__main__":
    app()
