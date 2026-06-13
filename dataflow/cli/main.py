"""CLI entry point for DataFlow-CV."""

import click
from pathlib import Path


def print_version(ctx, param, value):
    """Callback function: display version information"""
    if not value or ctx.resilient_parsing:
        return
    from dataflow import __version__
    click.echo(f"dataflow-cv version {__version__}")
    ctx.exit()


@click.group(context_settings={
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
    "show_default": True,
})
@click.option(
    "--version",
    "-v",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=print_version,
    help="Display version information",
)
@click.pass_context
def cli(ctx):
    """DataFlow-CV command line tool - Computer vision dataset processing toolkit"""
    # Initialize context object (default values)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = False
    ctx.obj["log_dir"] = Path("./logs")
    ctx.obj["strict"] = True
    # No logger creation — logging is module-owned


# Register subcommand groups
from .commands import convert, visualize, evaluate

cli.add_command(convert.convert_group)
cli.add_command(visualize.visualize_group)
cli.add_command(evaluate.evaluate_group)

if __name__ == "__main__":
    cli()