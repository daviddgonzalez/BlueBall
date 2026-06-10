"""Blue Ball — single entry point. See `python main.py -h` for subcommands.

    python main.py                 # play the game (default)
    python main.py watch           # watch the GA train, live
    python main.py train gym       # headless GA training (infinite|levels|maze|gym)
    python main.py repro-boost      # reproduce the boost-pad bug
    python main.py play-gym box-lava   # play one gym segment by hand
"""

import sys

from blueball.cli import main

if __name__ == "__main__":
    sys.exit(main())
