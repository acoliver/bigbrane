# Offline reproduction

Run from `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`. The builder reads only existing sources and writes only v2. Rebuilding repeats the bounded local selection; it does not acquire evidence.

```sh
export PYTHONDONTWRITEBYTECODE=1 BIGBRANE_LIVE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition/preparation/inventory/v2
export TMPDIR=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition/preparation/inventory/v2/tmp
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/v2/build_delta.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/v2/finalize_delta.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B -m pytest -c results-wt/heldout-acquisition/preparation/inventory/v2/pytest.ini --confcutdir=results-wt/heldout-acquisition/preparation/inventory/v2 --basetemp=results-wt/heldout-acquisition/preparation/inventory/v2/test-work --junitxml=results-wt/heldout-acquisition/preparation/inventory/v2/tests.xml results-wt/heldout-acquisition/preparation/inventory/v2/test_delta.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/v2/build_delta.py --validate
```

The original PDF remains at `results-wt/evidence-recovery/remaining-sources/mitchell-ch2.pdf`. The retained 18-line extraction is its first page, produced by local `pdftotext -layout`; only the footer is used as a title binding. No OCR model or document fetch is needed.
