import click
import os
import numpy as np
import netCDF4
from scipy.spatial import KDTree
from scipy.signal import convolve2d

import time
import spectral_util.spec_io as spec_io
from osgeo import osr
import pyproj
import logging
from tqdm import tqdm
from copy import deepcopy
osr.UseExceptions()

LAYERS = ('data', 'loc', 'glt')


def emit_location_layer(nc_file: str, layer: str):
    """Read the location group of an EMIT NetCDF file (radiance, observation,
    reflectance or mask) as (meta, array of rows x cols x bands).

    layer:
        loc  longitude, latitude, elevation, in that order (the band order ISOFIT
             and the EMIT LOC product use), on the instrument grid
        glt  the geometry lookup table as sample, line (1-based; 0 = no data), on
             the orthorectified grid described by the file's geotransform
    """
    with netCDF4.Dataset(nc_file) as ds:
        if 'location' not in ds.groups:
            raise ValueError(f'{nc_file} has no location group')
        location = ds['location']
        trans, proj = ds.geotransform, ds.spatial_ref
        if layer == 'glt':
            glt = np.stack([np.ma.getdata(location['glt_x'][:]), np.ma.getdata(location['glt_y'][:])], axis=-1)
            return spec_io.GenericGeoMetadata(['GLT Sample Lookup', 'GLT Line Lookup'], trans, proj,
                                              nodata_value=0), glt.astype(np.int32)
        if layer == 'loc':
            names = ['lon', 'lat', 'elev']
            loc = np.stack([np.ma.filled(location[n][:], location[n]._FillValue) for n in names], axis=-1)
            return spec_io.GenericGeoMetadata([location[n].long_name for n in names], trans, proj,
                                              nodata_value=float(location['lon']._FillValue)), loc
    raise ValueError(f'Unknown location layer {layer!r}; choose loc or glt')


@click.command()
@click.argument('nc_file')
@click.argument('output_file')
@click.option('--ortho', is_flag=True)
@click.option('--overwrite', is_flag=True)
@click.option('--layer', type=click.Choice(LAYERS), default='data', show_default=True,
              help='data: the main variable; loc: longitude, latitude, elevation; glt: geometry lookup table '
                   '(loc and glt: EMIT files)')
@click.option('--mask-type', type=click.Choice(['mask', 'band_mask']), default=None,
              help='for an EMIT L2A mask file: which mask variable to convert')
def nc_to_envi(nc_file: str, output_file: str, ortho=False, overwrite=False, layer='data', mask_type=None):
    """Convert a NetCDF file to ENVI format.

    \b
    Args:
        nc_file (str): Path to the input NetCDF file.
        output_file (str): Path to the output ENVI file.
        ortho (bool, optional): Whether to orthorectify the data. Defaults to False.
        layer (str, optional): data, loc or glt. Defaults to data.
        mask_type (str, optional): mask or band_mask, for EMIT L2A mask files.
    """
    if os.path.isfile(output_file) and not overwrite:
        raise FileExistsError(f'Output file {output_file} already exists. Use --overwrite to overwrite it.')

    if layer == 'glt':
        if ortho:
            raise click.UsageError('The GLT is already on the orthorectified grid; leave out --ortho.')
        meta, dat = emit_location_layer(nc_file, 'glt')
        spec_io.write_envi_file(dat, meta, output_file)
        return

    if layer == 'loc':
        meta, dat = emit_location_layer(nc_file, 'loc')
        if ortho:
            glt = emit_location_layer(nc_file, 'glt')[1]
            dat = spec_io.ortho_data(dat, glt, nodata_value=meta.nodata_value)
        spec_io.write_envi_file(dat, meta, output_file)
        return

    meta, dat = spec_io.load_data(nc_file, load_glt=True, mask_type=mask_type)
    dat = np.array(dat)

    if ortho and meta.orthoable:
        dat = spec_io.ortho_data(dat, meta.glt)

    spec_io.write_envi_file(dat, meta, output_file)





@click.group()
def cli():
    pass


cli.add_command(nc_to_envi)

if __name__ == '__main__':
    cli()
