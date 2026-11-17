"""spectral_util download emit-fid: URLs from the acquisition ID, and safe downloads."""

import os

import numpy as np
import pytest
from click.testing import CliRunner
from spectral.io import envi

from spectral_util.ea_assist import emit_fid
from spectral_util.spec_io import reformat

from emit_fixtures import BANDS, COLS, GEOTRANSFORM, ROWS, FakeDaac, make_obs, make_rad, write_netrc

FID = 'emit20240301t014122'
RAD = 'EMIT_L1B_RAD_002_20240301T014122.nc'
OBS = 'EMIT_L1B_OBS_002_20240301T014122.nc'


def test_urls_follow_from_the_acquisition_id():
    urls = emit_fid.l1b_urls(FID)
    base = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.002/EMIT_L1B_RAD_002_20240301T014122'
    assert urls == {'rad': f'{base}/{RAD}', 'obs': f'{base}/{OBS}'}
    for name in ('EMIT_L1B_RAD_001_20240301T014122_2406106_003', 'EMIT20240301T014122', FID.upper()):
        assert emit_fid.l1b_urls(name) == urls
    assert emit_fid.l1b_urls(FID, ['obs']) == {'obs': urls['obs']}
    for bad in ('', 'emit', 'emit20240301', '../emit20240301t014122', 'emit20240301t014122;id',
                'emit20240301t014122_20240302t010101', 'x20240301t014122'):
        with pytest.raises(ValueError):
            emit_fid.l1b_urls(bad)
    with pytest.raises(ValueError):
        emit_fid.l1b_urls(FID, ['rfl'])


@pytest.fixture
def daac(tmp_path):
    rad, obs = make_rad(tmp_path / 'src_rad.nc'), make_obs(tmp_path / 'src_obs.nc')
    d = FakeDaac({RAD: rad.read_bytes(), OBS: obs.read_bytes()})
    yield d
    d.close()


def session_for(daac, tmp_path, **kw):
    return emit_fid.earthdata_session(write_netrc(tmp_path / 'netrc', daac.urs_host, **kw), urs_host=daac.urs_host)


def test_download_through_earthdata_login(daac, tmp_path):
    out = tmp_path / 'out'
    paths = emit_fid.download_emit_fid(FID, str(out), base_url=daac.base_url, session=session_for(daac, tmp_path))
    assert paths == {'rad': str(out / RAD), 'obs': str(out / OBS)}
    assert (out / RAD).read_bytes() == daac.files[RAD]
    assert sorted(p.name for p in out.iterdir()) == sorted([RAD, OBS])          # no temporary files left
    # The credentials went to Earthdata Login and nowhere else.
    assert {role for role, auth in daac.auth_seen if auth} == {'urs'}
    # Already there: not downloaded again.
    n = len(daac.auth_seen)
    emit_fid.download_emit_fid(FID, str(out), base_url=daac.base_url, session=session_for(daac, tmp_path))
    assert len(daac.auth_seen) == n


def test_failures_leave_nothing_behind(daac, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()

    def attempt(fid=FID, **kw):
        with pytest.raises(emit_fid.DownloadError) as e:
            emit_fid.download_emit_fid(fid, str(out), base_url=daac.base_url, session=session_for(daac, tmp_path, **kw))
        assert list(out.iterdir()) == []
        return str(e.value)

    assert 'not found' in attempt('emit20250607t222745')                        # no V002 granule
    assert 'refused the credentials' in attempt(password='wrong')
    daac.truncate.add(RAD)
    assert 'incomplete' in attempt() or "couldn't download" in attempt()
    daac.truncate.clear()
    daac.html_instead = True
    assert 'web page' in attempt()
    daac.html_instead = False
    daac.files[RAD] = b'<?xml version="1.0"?><Error>not a file</Error>'
    assert 'NetCDF-4' in attempt()


def test_credentials_never_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))                                    # no ~/.netrc
    with pytest.raises(emit_fid.DownloadError, match='netrc'):
        emit_fid.earthdata_session()
    (tmp_path / 'other').write_text('machine example.org login a password b\n')
    with pytest.raises(emit_fid.DownloadError, match='no login and password'):
        emit_fid.earthdata_session(str(tmp_path / 'other'))
    assert not (tmp_path / '.netrc').exists()


def test_cli(daac, tmp_path, monkeypatch):
    monkeypatch.setattr(emit_fid, 'V002_L1B_BASE', daac.base_url)
    monkeypatch.setattr(emit_fid, 'URS_HOST', daac.urs_host)
    monkeypatch.setattr(emit_fid.download_emit_fid, '__defaults__',
                        (('rad', 'obs'), None, daac.base_url, False, None))
    monkeypatch.setattr(emit_fid.earthdata_session, '__defaults__', (None, daac.urs_host))
    netrc = write_netrc(tmp_path / 'netrc', daac.urs_host)
    r = CliRunner().invoke(emit_fid.emit_fid, [FID, str(tmp_path / 'out'), '--files', 'obs', '--netrc', str(netrc)])
    assert r.exit_code == 0, r.output
    assert r.output.strip() == str(tmp_path / 'out' / OBS)
    r = CliRunner().invoke(emit_fid.emit_fid, ['emit20250607t222745', str(tmp_path / 'out'), '--netrc', str(netrc)])
    assert r.exit_code != 0 and 'not found' in r.output


# --- reformat: radiance, observation, location and GLT as ENVI -------------
def test_reformat_layers(tmp_path):
    rad, obs = make_rad(tmp_path / RAD), make_obs(tmp_path / OBS)
    runner = CliRunner()

    def convert(src, name, *opts):
        r = runner.invoke(reformat.nc_to_envi, [str(src), str(tmp_path / name), *opts])
        assert r.exit_code == 0, (r.output, r.exception)
        img = envi.open(str(tmp_path / f'{name}.hdr'))
        return img, np.array(img.open_memmap(interleave='bip'))

    img, data = convert(rad, 'rdn')
    assert data.shape == (ROWS, COLS, BANDS) and data[1, 2, 3] == (1 * COLS + 2) * BANDS + 3
    assert [float(w) for w in img.metadata['wavelength']] == [400, 500, 600, 700]

    img, data = convert(obs, 'obs')
    assert data.shape == (ROWS, COLS, 11) and img.metadata['band names'][0] == 'path length'

    img, data = convert(obs, 'loc', '--layer', 'loc')
    assert data.shape == (ROWS, COLS, 3) and data.dtype == np.float64
    assert img.metadata['band names'] == ['Longitude (WGS-84)', 'Latitude (WGS-84)', 'Surface Elevation']
    assert np.allclose(data[0, 1], [-117.999, 34.001, 100.001])

    img, data = convert(rad, 'glt', '--layer', 'glt')
    assert data.shape == (ROWS + 1, COLS + 1, 2) and data.dtype == np.int32
    assert img.metadata['band names'] == ['GLT Sample Lookup', 'GLT Line Lookup']
    assert data[2, 3].tolist() == [4, 3] and data[ROWS, COLS].tolist() == [0, 0]
    assert img.metadata['data ignore value'] == '0'
    assert [float(v) for v in img.metadata['map info'][3:7]][:2] == [GEOTRANSFORM[0], GEOTRANSFORM[3]]

    r = runner.invoke(reformat.nc_to_envi, [str(rad), str(tmp_path / 'glt'), '--layer', 'glt'])
    assert r.exit_code != 0                                                     # won't overwrite unasked
    r = runner.invoke(reformat.nc_to_envi, [str(rad), str(tmp_path / 'glt2'), '--layer', 'glt', '--ortho'])
    assert r.exit_code != 0
