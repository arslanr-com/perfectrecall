"""Content-free diagnostics for the active PerfectRecall runtime."""
import os
import sys
from perfectrecall import __version__
from mnemosyne.core import jev

def collect_runtime_diagnostics():
    settings = jev.settings()
    ready = bool(os.environ.get(settings['key_env'], '').strip())
    checks = [
        dict(category='env', check='python_version', status='OK', detail=sys.version.split()[0]),
        dict(category='package', check='perfectrecall_version', status='OK', detail=__version__),
        dict(category='core', check='recall_backend', status='OK', detail='Jev decisions'),
        dict(category='core', check='api_key_configured', status='OK' if ready else 'MISSING',
             detail='configured' if ready else 'Set ' + settings['key_env']),
    ]
    return dict(status='ok' if ready else 'warning', checks=checks)
