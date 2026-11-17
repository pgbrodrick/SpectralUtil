"""Small synthetic EMIT L1B files and a fake LP DAAC, for tests."""

import base64
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import netCDF4
import numpy as np

ROWS, COLS, BANDS = 6, 5, 4
GEOTRANSFORM = [-118.0, 0.001, 0.0, 34.0, 0.0, -0.001]
WGS84 = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
         'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


def _location(ds):
    loc = ds.createGroup('location')
    ds.createDimension('ortho_y', ROWS + 1)
    ds.createDimension('ortho_x', COLS + 1)
    for name, long_name, base in (('lon', 'Longitude (WGS-84)', -118.0), ('lat', 'Latitude (WGS-84)', 34.0),
                                  ('elev', 'Surface Elevation', 100.0)):
        v = loc.createVariable(name, 'f8', ('downtrack', 'crosstrack'), fill_value=-9999.0)
        v.long_name = long_name
        v[:] = base + np.arange(ROWS * COLS).reshape(ROWS, COLS) * 0.001
    gx = loc.createVariable('glt_x', 'i4', ('ortho_y', 'ortho_x'), fill_value=0)
    gy = loc.createVariable('glt_y', 'i4', ('ortho_y', 'ortho_x'), fill_value=0)
    glt_x = np.zeros((ROWS + 1, COLS + 1), np.int32)
    glt_y = np.zeros((ROWS + 1, COLS + 1), np.int32)
    glt_x[:ROWS, :COLS] = np.arange(1, COLS + 1)[None, :]
    glt_y[:ROWS, :COLS] = np.arange(1, ROWS + 1)[:, None]
    gx[:] = glt_x
    gy[:] = glt_y
    ds.geotransform = GEOTRANSFORM
    ds.spatial_ref = WGS84


def make_rad(path):
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('downtrack', ROWS)
        ds.createDimension('crosstrack', COLS)
        ds.createDimension('bands', BANDS)
        v = ds.createVariable('radiance', 'f4', ('downtrack', 'crosstrack', 'bands'), fill_value=-9999.0)
        v[:] = np.arange(ROWS * COLS * BANDS, dtype=np.float32).reshape(ROWS, COLS, BANDS)
        sbp = ds.createGroup('sensor_band_parameters')
        sbp.createVariable('wavelengths', 'f4', ('bands',))[:] = [400.0, 500.0, 600.0, 700.0]
        sbp.createVariable('fwhm', 'f4', ('bands',))[:] = [8.0, 8.0, 8.0, 8.0]
        _location(ds)
    return path


def make_obs(path):
    names = ['path length', 'to-sensor azimuth', 'to-sensor zenith', 'to-sun azimuth', 'to-sun zenith',
             'solar phase', 'slope', 'aspect', 'cosine(i)', 'UTC time', 'earth-sun distance']
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('downtrack', ROWS)
        ds.createDimension('crosstrack', COLS)
        ds.createDimension('bands', len(names))
        v = ds.createVariable('obs', 'f8', ('downtrack', 'crosstrack', 'bands'), fill_value=-9999.0)
        v[:] = np.arange(ROWS * COLS * len(names), dtype=np.float64).reshape(ROWS, COLS, len(names))
        sbp = ds.createGroup('sensor_band_parameters')
        ob = sbp.createVariable('observation_bands', str, ('bands',))
        for i, n in enumerate(names):
            ob[i] = n
        _location(ds)
    return path


class FakeDaac:
    """Three servers on loopback, imitating how the LP DAAC serves protected files:

    data host:  /<collection>/<granule>/<file>  -> without a session cookie, a
                redirect to Earthdata Login; with one, a redirect to the file store
    urs host:   /oauth/authorize  needs HTTP Basic credentials, then redirects back
                to the data host's /callback, which sets the session cookie
    store host: serves the file, and refuses any request carrying credentials
                (as S3 does with signed URLs)
    """

    def __init__(self, files, username='user', password='pass'):
        self.files = files                  # {file name: bytes}
        self.username, self.password = username, password
        self.auth_seen = []                 # (host role, Authorization header) for every request
        self.truncate = set()               # file names to cut short
        self.html_instead = False           # answer data requests with a login page
        self.servers = {}
        for role in ('data', 'urs', 'store'):
            srv = ThreadingHTTPServer(('127.0.0.1', 0), self._handler(role))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.servers[role] = srv

    def url(self, role):
        return f'http://127.0.0.1:{self.servers[role].server_address[1]}'

    @property
    def urs_host(self):
        return f'127.0.0.1:{self.servers["urs"].server_address[1]}'

    @property
    def base_url(self):
        return self.url('data') + '/lp-prod-protected/EMITL1BRAD.002'

    def close(self):
        for srv in self.servers.values():
            srv.shutdown()
            srv.server_close()

    def _handler(self, role):
        daac = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, status, headers=(), body=b''):
                self.send_response(status)
                for k, v in headers:
                    self.send_header(k, v)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                daac.auth_seen.append((role, self.headers.get('Authorization')))
                u = urllib.parse.urlsplit(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                if role == 'urs':
                    want = 'Basic ' + base64.b64encode(f'{daac.username}:{daac.password}'.encode()).decode()
                    if self.headers.get('Authorization') != want:
                        return self.send(401, [('Content-Type', 'text/html')], b'<html>login</html>')
                    target = daac.url('data') + '/callback?' + urllib.parse.urlencode({'next': q.get('next', '/')})
                    return self.send(302, [('Location', target)])
                if role == 'data':
                    if u.path == '/callback':
                        return self.send(302, [('Location', daac.url('data') + q['next']),
                                               ('Set-Cookie', 'daac_session=ok; Path=/')])
                    if daac.html_instead:
                        return self.send(200, [('Content-Type', 'text/html')], b'<html>please log in</html>')
                    if 'daac_session=ok' not in (self.headers.get('Cookie') or ''):
                        target = daac.url('urs') + '/oauth/authorize?' + urllib.parse.urlencode({'next': u.path})
                        return self.send(302, [('Location', target)])
                    name = u.path.rsplit('/', 1)[-1]
                    if name not in daac.files:
                        return self.send(404, [('Content-Type', 'application/xml')], b'<Error>NoSuchKey</Error>')
                    return self.send(302, [('Location', daac.url('store') + '/' + name + '?X-Amz-Signature=x')])
                # store
                if self.headers.get('Authorization'):
                    return self.send(400, [('Content-Type', 'application/xml')], b'<Error>two auth methods</Error>')
                name = u.path.lstrip('/')
                body = daac.files.get(name)
                if body is None:
                    return self.send(404)
                if name in daac.truncate:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-netcdf')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.send(200, [('Content-Type', 'application/x-netcdf')], body)

        return Handler


def write_netrc(path, host, username='user', password='pass'):
    with open(path, 'w') as f:
        f.write(f'machine {host.split(":")[0]} login {username} password {password}\n')
    os.chmod(path, 0o600)
    return path
