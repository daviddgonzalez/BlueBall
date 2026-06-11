# Blue Ball

A Red Ball-inspired 2D physics platformer built with PyGame and Pymunk.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    python main.py

## Commands

`main.py` is the single entry point — `python main.py -h` lists everything.

    python main.py                 # play the game (default)
    python main.py watch           # watch the GA train, live
    python main.py train infinite  # headless GA training:
    python main.py train levels    #   infinite | levels | maze | gym
    python main.py train maze      #   (each takes -h for its flags)
    python main.py train gym
    python main.py repro-boost      # reproduce the boost-pad bug (headless)
    python main.py repro-boost --play   # ...as a playable level

## Running tests

    pytest -q
