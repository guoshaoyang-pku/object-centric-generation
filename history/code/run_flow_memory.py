"""Run the unmodified flow CLI and persist process-local CUDA memory peaks."""
import json
import sys
import time
from pathlib import Path

import torch
import codec_flow_research


if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    started = time.time()
    outcome = 'completed'
    try:
        codec_flow_research.main()
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        record = {'started_unix': started, 'finished_unix': time.time(),
                  'outcome': outcome, 'scope': 'This process only, includes setup/one-step/save',
                  'argv': sys.argv[1:]}
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        out.mkdir(parents=True, exist_ok=True)
        (out / 'batch_memory_admission.json').write_text(json.dumps(record, indent=2) + '\n')
