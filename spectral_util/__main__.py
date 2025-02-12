import importlib

import click


class LazyCLI(click.Group):
    """
    Custom click class to load commands at runtime.

    Reference: https://click.palletsprojects.com/en/stable/complex/#defining-the-lazy-group
    """
    def __init__(self, *args, lazy_subcommands={}, **kwargs):
        super().__init__(*args, **kwargs)

        self.lazy_subcommands = lazy_subcommands

    def _lazy_load(self, cmd_name):
        try:
            module, func = self.lazy_subcommands[cmd_name].split(":")
            module = importlib.import_module(module)
            return getattr(module, func)
        except Exception as e:
            print(f"Failed to import {cmd_name!r}, reason: {e}")

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = list(self.lazy_subcommands)
        return base + lazy

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)


@click.group(
    cls=LazyCLI,
    no_args_is_help=True,
    lazy_subcommands={
        "glt": "spectral_util.glt:cli",
        "ndvi": "spectral_util.utils.spectral_util:ndvi",
        "nbr": "spectral_util.utils.spectral_util:nbr",
        "rgb": "spectral_util.utils.spectral_util:rgb",
        # "ndwi": "specio.utils.spectral_util:ndwi"
    },
)
def cli(**kwargs):
    """\
    Spectral Utilities

    \b
    Repository: https://github.com/emit-sds/SpectralUtil
    Report an issue: https://github.com/emit-sds/SpectralUtil/issues
    """
    pass


if __name__ == "__main__":
    cli()
