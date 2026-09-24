# /// script
# dependencies = ["pymunk", "numpy"]
# ///
"""Blue Ball — single entry point. See `python main.py -h` for subcommands.

    python main.py                 # play the game (default)
    python main.py watch           # watch the GA train, live
    python main.py train gym       # headless GA training (infinite|levels|maze|gym)
    python main.py repro-boost      # reproduce the boost-pad bug
    python main.py play-gym box-lava   # play one gym segment by hand

In the browser (pygbag) there is no install step, so src/ must be put on
sys.path here, and the game runs through the async play loop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if sys.platform == "emscripten":
    # pygbag scans main.py's own imports to set up the browser runtime, so
    # pygame must be imported here, not only inside the blueball package.
    import asyncio

    import pygame  # noqa: F401

    from blueball.web import play_web

    asyncio.run(play_web())
elif __name__ == "__main__":
    from blueball.cli import main

    sys.exit(main())
