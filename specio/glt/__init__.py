import click

from specio.__main__ import LazyCLI


@click.group(
    cls=LazyCLI,
    no_args_is_help=True,
    lazy_subcommands={
        "build": "specio.glt.mosaic:build_obs_nc",
        "apply": "specio.glt.mosaic:apply_glt",
    },
)
def cli(**kwargs):
    """\
    GLT Mosaicing functions
    """
    pass
