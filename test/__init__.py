# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0

import os

if os.getenv("GITHUB_ACTIONS") or os.getenv("ENABLE_COREDUMPY"):
    try:
        import coredumpy
        coredumpy.patch_pytest(directory=os.getenv("COREDUMPY_DUMP_DIR", "./"))
    except ImportError:
        pass
