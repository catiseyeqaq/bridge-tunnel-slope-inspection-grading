"""Restore the frozen PPU runtime using separately provisioned model assets.

The upload contains code and offline Python wheels, not model weights.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matches(path, item):
    return (path.is_file() and path.stat().st_size == item['bytes']
            and digest(path) == item['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('~/track4_current/venv'))
    parser.add_argument('--local-root', type=Path, default=Path('~/track4_current'))
    parser.add_argument('--asset-root', type=Path, action='append', default=[],
                        help='Separately provisioned complete release root, containing code/ and model_assets/')
    parser.add_argument('--verify-only', action='store_true', help='Validate sources; do not copy or install')
    parser.add_argument('--stage-only', action='store_true', help='Restore files without installing a venv')
    args = parser.parse_args()
    if args.verify_only and args.stage_only:
        parser.error('Choose only one of --verify-only and --stage-only')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use /usr/local/bin/python3 from the official Python 3.12 PPU image')
    root = Path(__file__).resolve().parent
    local = args.local_root.resolve()
    release = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    external = json.loads((root / 'EXTERNAL_MODEL_MANIFEST.json').read_text())
    stores = args.asset_root + [Path('~/track4/assets/p2_runtime_assets'), Path('~/track4_current')]
    plan = []
    missing = []
    for name, item in release.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe release path: ' + name)
        destination = local / relative
        source = root / relative
        if source.is_file():
            if not matches(source, item):
                raise RuntimeError('Damaged packaged file: ' + name)
        elif name in external:
            candidates = [destination] + [store / relative for store in stores]
            source = next((candidate for candidate in candidates if matches(candidate, item)), None)
            if source is None:
                missing.append(name)
                continue
        else:
            raise FileNotFoundError('Missing packaged code/dependency: ' + name)
        plan.append((name, item, source, destination))
    if missing:
        raise FileNotFoundError(
            'Model weights are intentionally NOT in the upload. Provision the exact assets separately '
            'and pass --asset-root RELEASE_ROOT; required missing/mismatched assets: ' + ', '.join(missing))
    runtime = root / 'offline_runtime'
    manifest = json.loads((runtime / 'RUNTIME_MANIFEST.json').read_text())
    native = Path('/usr/local/lib/python3.12/site-packages')
    for item in manifest['provided']:
        if md.version(item['name']) != item['version']:
            raise RuntimeError('Official PPU native version mismatch: ' + item['name'])
        for name in item['top_level_entries']:
            path = Path(item['site']) / name
            if not path.exists() or not path.resolve().is_relative_to(native):
                raise RuntimeError('Official PPU native dependency missing: ' + name)
    if args.verify_only:
        print(json.dumps({'verified': True, 'restored': False, 'installed': False,
                          'release_files': len(plan), 'external_model_files': len(external),
                          'weights_in_upload': False, 'local_root': str(local)}))
        return
    if not args.stage_only and args.venv.exists():
        raise FileExistsError('Existing environments are never overwritten: ' + str(args.venv))
    for name, item, source, destination in plan:
        if source.resolve() == destination.resolve() or matches(destination, item):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.restoring')
        shutil.copyfile(source, temporary)
        if not matches(temporary, item):
            raise RuntimeError('Release restoration mismatch: ' + name)
        temporary.replace(destination)
    local.mkdir(parents=True, exist_ok=True)
    if root != local:
        shutil.copyfile(root / 'RELEASE_MANIFEST.json', local / 'RELEASE_MANIFEST.json')
    (local / 'run.sh').chmod(0o755)
    if args.stage_only:
        print(json.dumps({'restored': True, 'installed': False, 'local_root': str(local),
                          'release_files': len(plan), 'external_model_files': len(external)}))
        return
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(args.venv)
    site = args.venv / 'lib/python3.12/site-packages'
    for item in manifest['provided']:
        for name in item['top_level_entries']:
            source = Path(item['site']) / name
            (site / name).symlink_to(source, target_is_directory=source.is_dir())
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('TRACK4_') or name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
            env.pop(name)
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [str(args.venv / 'bin/python'), '-m', 'pip', 'install', '--no-index',
               '--no-deps', '--disable-pip-version-check',
               *[str(runtime / item['file']) for item in manifest['packages']]]
    subprocess.run(command, env=env, check=True)
    print(json.dumps({'installed': True, 'venv': str(args.venv),
                      'offline_wheels': len(manifest['packages']),
                      'native_image_packages': [x['name'] for x in manifest['provided']],
                      'system_site_packages': False, 'weights_in_upload': False,
                      'external_model_files': len(external)}))


if __name__ == '__main__':
    main()
