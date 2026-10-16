"""P0 probe 8: run the trusted platform-trial fixture and write its full record."""
import json
import sys

sys.path.insert(0, "/in")
import numpy

from trial_fixture import CANDIDATES, evaluate

record = evaluate(CANDIDATES[0])
text = json.dumps(record, sort_keys=True, allow_nan=False)
open("/out/record.json", "w").write(text)
json.dump({"numpy": numpy.__version__, "hmm_kl_hex": record["metrics"]["hmm_kl"].hex(),
           "rnn_kl_hex": record["metrics"]["rnn_kl"].hex()}, open("/out/result.json", "w"))
