import json
import os
import shutil
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
results = root / 'results-wt/seed-provenance'
python = '/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python'
isolated = results / 'isolated'
isolated.mkdir()
tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files']).decode().splitlines()
for name in tracked:
    source = root / name
    if source.is_file():
        destination = isolated / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
for name in ['hybrid/literature/seed_provenance.json', 'hybrid/tests/literature/test_seed_provenance.py']:
    destination = isolated / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, destination)
assert not (isolated / 'hybrid/literature/cache').exists()
assert (isolated / 'hybrid/tests/literature/test_seed.py').read_bytes() == (root / 'hybrid/tests/literature/test_seed.py').read_bytes()
records = []
lanes = [('isolated', isolated, ['hybrid/tests/literature/test_seed.py', 'hybrid/tests/literature/test_seed_provenance.py']), ('literature', root, ['hybrid/tests/literature'])]
for label, cwd, tests in lanes:
    env = dict(os.environ, BIGBRANE_LIVE='0', PYTHONPATH=str(results / 'offline') + os.pathsep + str(cwd))
    command = [python, '-m', 'pytest'] + tests + ['-q', '--junitxml=' + str(results / (label + '.xml'))]
    print(label, 'starting', flush=True)
    output = (results / (label + '.log')).open('w')
    run = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    output.close()
    records.append(dict(label=label, cwd=str(cwd), command=command, returncode=run.returncode, PYTHONPATH=env['PYTHONPATH']))
    print(label, 'finished', run.returncode, flush=True)
assert not (isolated / 'hybrid/literature/cache').exists()
(results / 'verification.json').write_text(json.dumps(records, indent=2))
print('cache absent before and after isolated run', flush=True)
