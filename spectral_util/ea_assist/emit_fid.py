"""Download EMIT L1B files for one acquisition, from URLs built from its ID.

V002 L1B granules are named only by their acquisition time
(EMIT_L1B_RAD_002_20240301T014122), so the URL of each file follows from the
acquisition ID without a search:

    <base>/EMIT_L1B_RAD_002_<time>/EMIT_L1B_RAD_002_<time>.nc
    <base>/EMIT_L1B_RAD_002_<time>/EMIT_L1B_OBS_002_<time>.nc

Meant to run unattended (batch jobs):

- Credentials come only from a netrc file (an entry for urs.earthdata.nasa.gov),
  by default ~/.netrc. Nothing ever prompts, and nothing is written to the home
  directory. The credentials are sent only to Earthdata Login, never to the
  data host or wherever it redirects.
- Each file is downloaded to a temporary name beside its destination, checked
  (HTTP status, size against Content-Length, and that it starts like a
  NetCDF-4/HDF5 file rather than a login page), and only then renamed into
  place. So a file under its final name is always complete; an interrupted
  download never looks finished.
- Any failure raises (and the command exits non-zero).

    spectral_util download emit-fid emit20240301t014122 OUTDIR
"""

from __future__ import annotations

import netrc
import os
import re
import urllib.parse

import click
import requests

V002_L1B_BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.002"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
URS_HOST = "urs.earthdata.nasa.gov"
FILES = {"rad": "EMIT_L1B_RAD_002_{t}.nc", "obs": "EMIT_L1B_OBS_002_{t}.nc"}
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_TIME = re.compile(r"(?<!\d)(\d{8})[tT](\d{6})(?!\d)")


class DownloadError(Exception):
    pass


def acquisition_time(fid: str) -> str:
    """'20240301T014122' from an acquisition ID (emit20240301t014122) or a
    granule name containing exactly one such timestamp."""
    if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_.\-]{1,100}", fid.strip()) \
            or not fid.strip().lower().startswith("emit"):
        raise ValueError(f"not an EMIT acquisition ID: {fid!r}")
    found = _TIME.findall(fid)
    if len(found) != 1:
        raise ValueError(f"not an EMIT acquisition ID: {fid!r}")
    return f"{found[0][0]}T{found[0][1]}"


def l1b_urls(fid: str, files=("rad", "obs"), base_url: str = V002_L1B_BASE) -> dict:
    """{file key: URL} for the V002 L1B files of an acquisition."""
    t = acquisition_time(fid)
    unknown = set(files) - set(FILES)
    if unknown:
        raise ValueError(f"unknown file(s) {sorted(unknown)}; choose from {sorted(FILES)}")
    granule = f"EMIT_L1B_RAD_002_{t}"
    return {k: f"{base_url.rstrip('/')}/{granule}/{FILES[k].format(t=t)}" for k in files}


def v002_radiance_exists(fid: str, timeout: float = 30) -> bool:
    """Whether the LP DAAC lists V002 radiance for this acquisition (NASA's CMR
    catalog; no login needed)."""
    r = requests.get(CMR_GRANULES, timeout=timeout, params={
        "short_name": "EMITL1BRAD", "version": "002", "page_size": 0,
        "granule_ur": f"EMIT_L1B_RAD_002_{acquisition_time(fid)}"})
    r.raise_for_status()
    return int(r.headers.get("CMR-Hits", "0")) > 0


def _same_host(url: str, host: str) -> bool:
    u = urllib.parse.urlsplit(url)
    netloc = (u.hostname or "") + (f":{u.port}" if u.port and u.port != 443 else "")
    return u.scheme in ("https", "http") and netloc.lower() == host.lower()


class EarthdataSession(requests.Session):
    """A requests session that sends the netrc credentials to Earthdata Login
    only: requests drops credentials when a redirect leaves a host, and this
    adds them back only for the Earthdata Login host."""

    def __init__(self, username: str, password: str, urs_host: str = URS_HOST):
        super().__init__()
        self._credentials = (username, password)
        self._urs_host = urs_host

    def rebuild_auth(self, prepared_request, response):
        prepared_request.headers.pop("Authorization", None)
        if _same_host(prepared_request.url, self._urs_host):
            prepared_request.prepare_auth(self._credentials)


def earthdata_session(netrc_path: str | None = None, urs_host: str = URS_HOST) -> EarthdataSession:
    """A session with the Earthdata Login credentials from `netrc_path` (default
    ~/.netrc). Raises DownloadError if there are none: never prompts."""
    path = netrc_path or os.path.join(os.path.expanduser("~"), ".netrc")
    try:
        entry = netrc.netrc(path).authenticators(urs_host.split(":")[0])
    except (OSError, netrc.NetrcParseError) as e:
        raise DownloadError(f"can't read Earthdata Login credentials from {path}: {e}") from None
    if not entry or not entry[0] or not entry[2]:
        raise DownloadError(f"{path} has no login and password for machine {urs_host.split(':')[0]}")
    session = EarthdataSession(entry[0], entry[2], urs_host)
    session.trust_env = True        # keep proxy settings from the environment
    return session


def download_file(session: requests.Session, url: str, dest: str, timeout: float = 300) -> str:
    """Download `url` to `dest` through a checked temporary file; returns `dest`."""
    tmp = f"{dest}.partial-{os.getpid()}"
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                raise DownloadError(f"not found at the DAAC: {url}")
            if r.status_code in (401, 403):
                raise DownloadError(f"the DAAC refused the credentials (HTTP {r.status_code}) for {url}")
            if r.status_code != 200:
                raise DownloadError(f"HTTP {r.status_code} for {url}")
            if "text/html" in r.headers.get("Content-Type", ""):
                raise DownloadError(f"got a web page instead of data for {url} (a login problem?)")
            expected = r.headers.get("Content-Length")
            written = 0
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 2**20):
                    f.write(chunk)
                    written += len(chunk)
                f.flush()
                os.fsync(f.fileno())
        if expected is not None and written != int(expected):
            raise DownloadError(f"incomplete download of {url}: {written} of {expected} bytes")
        with open(tmp, "rb") as f:
            if f.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
                raise DownloadError(f"{url} didn't return a NetCDF-4 file")
        os.replace(tmp, dest)
        return dest
    except requests.RequestException as e:
        raise DownloadError(f"couldn't download {url}: {type(e).__name__}: {e}") from None
    finally:
        if os.path.lexists(tmp):        # only ever our own temporary file
            os.unlink(tmp)


def download_emit_fid(fid: str, outdir: str, files=("rad", "obs"), netrc_path: str | None = None,
                      base_url: str = V002_L1B_BASE, overwrite: bool = False,
                      session: requests.Session | None = None) -> dict:
    """Download the V002 L1B `files` of acquisition `fid` into `outdir`;
    returns {file key: path}. A file already there under its final name is
    complete (see the module notes) and is kept unless `overwrite`."""
    urls = l1b_urls(fid, files, base_url)
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for key, url in urls.items():
        dest = os.path.join(outdir, url.rsplit("/", 1)[1])
        if overwrite or not os.path.isfile(dest):
            session = session or earthdata_session(netrc_path)
            download_file(session, url, dest)
        out[key] = dest
    return out


@click.command()
@click.argument("fid")
@click.argument("outdir", type=click.Path(file_okay=False))
@click.option("--files", default="rad,obs", show_default=True, help="comma-separated: rad, obs")
@click.option("--netrc", "netrc_path", type=click.Path(dir_okay=False), help="netrc file (default ~/.netrc)")
@click.option("--overwrite", is_flag=True, help="download again even if a file is already there")
def emit_fid(fid, outdir, files, netrc_path, overwrite):
    """Download the V002 L1B files (radiance, observation) of one EMIT acquisition.

    FID is an acquisition ID (emit20240301t014122) or a granule name.
    """
    try:
        paths = download_emit_fid(fid, outdir, [f.strip() for f in files.split(",") if f.strip()],
                                  netrc_path=netrc_path, overwrite=overwrite)
    except (ValueError, DownloadError) as e:
        raise click.ClickException(str(e))
    for path in paths.values():
        click.echo(path)
