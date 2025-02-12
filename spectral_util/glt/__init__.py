import click

from spectral_util.__main__ import LazyCLI


@click.group(
    cls=LazyCLI,
    no_args_is_help=True,
    lazy_subcommands={
        "build": "spectral_util.glt.mosaic:build_obs_nc",
        "apply": "spectral_util.glt.mosaic:apply_glt",
    },
)
def cli(**kwargs):
    """\
    GLT Mosaicing functions
    """
    pass
