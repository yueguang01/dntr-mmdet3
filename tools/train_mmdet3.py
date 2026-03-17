import argparse
import os
import sys

# Python 3.12 removed stdlib distutils, but some MMEngine versions still import
# `distutils.errors` while collecting environment info.
try:
    import distutils  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types

    _distutils_errors = types.ModuleType('distutils.errors')

    class DistutilsError(Exception):
        pass

    class DistutilsModuleError(DistutilsError):
        pass

    class DistutilsClassError(DistutilsError):
        pass

    class DistutilsPlatformError(DistutilsError):
        pass

    _distutils_errors.DistutilsError = DistutilsError
    _distutils_errors.DistutilsModuleError = DistutilsModuleError
    _distutils_errors.DistutilsClassError = DistutilsClassError
    _distutils_errors.DistutilsPlatformError = DistutilsPlatformError

    _distutils_mod = types.ModuleType('distutils')
    _distutils_mod.errors = _distutils_errors
    sys.modules['distutils'] = _distutils_mod
    sys.modules['distutils.errors'] = _distutils_errors

from mmengine.config import Config
from mmengine.runner import Runner

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure custom modules are imported before building from cfg.
import dntr_custom  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='DNTR MMDet3 training launcher')
    parser.add_argument('config', help='Path to training config file')
    parser.add_argument('--work-dir', default=None, help='Override work_dir in config')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint in work_dir')
    parser.add_argument('--amp', action='store_true', help='Enable AMP training')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.amp:
        optim_wrapper = cfg.get('optim_wrapper', None)
        if optim_wrapper is None:
            raise RuntimeError('optim_wrapper is missing in config; cannot enable AMP automatically.')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'

    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
