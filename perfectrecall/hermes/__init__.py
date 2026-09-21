"""Hermes plugin entry point with PerfectRecall's defaults and legacy tool names."""
from perfectrecall import configure

configure()
from hermes_memory_provider import MnemosyneMemoryProvider  # noqa: E402


class PerfectRecallMemoryProvider(MnemosyneMemoryProvider):
    @property
    def name(self):
        return "perfectrecall"


def register_memory_provider(ctx):
    ctx.register_memory_provider(PerfectRecallMemoryProvider())


def register(ctx):
    # Register explicitly: class discovery can otherwise select the imported
    # compatibility base class before PerfectRecall alphabetically.
    if hasattr(ctx, "register_memory_provider"):
        register_memory_provider(ctx)
    from hermes_memory_provider.cli import register_cli, mnemosyne_command
    from perfectrecall import __version__

    def command(args):
        if getattr(args, "mnemosyne_cmd", None) == "version":
            print("PerfectRecall " + __version__)
            return 0
        return mnemosyne_command(args)

    ctx.register_cli_command(name="perfectrecall", help="Manage PerfectRecall memory",
        description="Inspect and consolidate memory using Jev decisions.",
        setup_fn=register_cli, handler_fn=command)
