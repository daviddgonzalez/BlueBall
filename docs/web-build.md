# Web build

BlueBall runs in the browser via [pygbag](https://pypi.org/project/pygbag/)
(pygame-ce compiled to WebAssembly), deployed to GitHub Pages by
`.github/workflows/deploy.yml` on every push to `master`.

    tools/build_web.sh            # clean build into build/web/
    tools/build_web.sh --serve    # build + serve on http://localhost:8000

Only the game itself ships: `main.py` takes a separate browser path through
`src/blueball/web.py`. Training, watch and the rest of the CLI are
native-only. The setup mirrors Gravi's (`~/projects/Gravi/docs/web-build.md`
has the longer write-up of the shared pygbag gotchas).

## Things that broke the first boot (2026-09-23)

- **pymunk version.** The browser only has pymunk 6.4; the desktop runs 7.x.
  BlueBall registers collisions with pymunk 7's `Space.on_collision` and
  `arbiter.process_collision`. `src/blueball/pymunk_compat.py` installs both
  on 6.x and does nothing on 7. The full test suite passes on pymunk 6.4 +
  numpy 1.26 as well as on the desktop versions. Keep it that way: a new
  pymunk-7-only call will pass tests locally and break only in the browser.
- **pygame must be imported in `main.py` itself.** With pygame imported only
  inside the `blueball` package, the browser crashed on
  `module 'pygame' has no attribute 'init'`. pygbag reads `main.py`'s own
  imports to decide what to install up front.
- **Don't import `blueball.cli` in the browser.** It pulls in
  `multiprocessing` and the training stack. `web.py` imports only what play
  needs.
- **Sound files are `.wav`,** which pygbag rejects by default. The build
  passes `--disable-sound-format-error`; browsers play `.wav` fine.
- **`pygbag.ini` keeps the rest of the repo out of the package.** Without it
  the build swept up `.claude/worktrees/` and `.venv/`.

## Debugging a grey screen

Grey after click-to-start means Python never reached the game. The browser
console won't show the Python error. Open `http://localhost:8000/#debug`
instead, which shows pygbag's terminal with the traceback. After any
rebuild, reload with Ctrl+Shift+R, or the browser keeps the old `.apk`.
