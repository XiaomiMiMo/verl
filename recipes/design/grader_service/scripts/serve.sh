#!/bin/bash
set -euo pipefail

GRADER=${GRADER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${LLM_JUDGE_API_KEY:?set LLM_JUDGE_API_KEY (the query + pick judges)}"

DESIGN_GRADER_ROOT=${DESIGN_GRADER_ROOT:?set DESIGN_GRADER_ROOT (our writable root)}
DESIGN_GRADER_ENV=${DESIGN_GRADER_ENV:?set DESIGN_GRADER_ENV (dir holding ms-playwright/ and fonts/fonts.conf)}

export PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH:-${DESIGN_GRADER_ENV}/ms-playwright}
export http_proxy=${http_proxy:-}
export https_proxy=${https_proxy:-}
export no_proxy=${no_proxy:-"localhost,127.0.0.1"}
export DESIGN_GRADER_ASSETS=${DESIGN_GRADER_ASSETS:-${DESIGN_GRADER_ROOT}/../assets}
export DESIGN_GRADER_WORK=${DESIGN_GRADER_WORK:-${DESIGN_GRADER_ROOT}/work/$(hostname)}
export DESIGN_GRADER_FONTCONF=${DESIGN_GRADER_FONTCONF:-${DESIGN_GRADER_ENV}/fonts/fonts.conf}

for _w in "${DESIGN_GRADER_ASSETS}" "${DESIGN_GRADER_WORK}"; do
    case "${_w}" in
        "${DESIGN_GRADER_ROOT%/}"/*|"${DESIGN_GRADER_ROOT%/}"/../assets*|/tmp/*) : ;;
        *) echo "[serve] FATAL: write path outside DESIGN_GRADER_ROOT: ${_w}" >&2; exit 1 ;;
    esac
done
for _ro in "${PLAYWRIGHT_BROWSERS_PATH}" "${DESIGN_GRADER_FONTCONF}"; do
    [ -e "${_ro}" ] || { echo "[serve] FATAL: read-only dep missing: ${_ro}" >&2
                         echo "        set \$DESIGN_GRADER_ENV to a prefix holding the" >&2
                         echo "        playwright browsers and the fontconfig dir" >&2
                         exit 1; }
done

ulimit -c 0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[serve] PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_BROWSERS_PATH"
echo "[serve] proxy=$http_proxy   no_proxy=$no_proxy"

echo "[serve] installing python deps"
pip install -q playwright pillow

EXPECT=$(python3 - <<'PY'
import json, os, playwright
loc = os.path.dirname(playwright.__file__)
for c in ("driver/browsers.json", "driver/package/browsers.json"):
    p = os.path.join(loc, c)
    if os.path.isfile(p):
        for b in json.load(open(p))["browsers"]:
            if b["name"] == "chromium-headless-shell":
                print(b["revision"]); break
        break
PY
)
SHELL_BIN="$PLAYWRIGHT_BROWSERS_PATH/chromium_headless_shell-${EXPECT}/chrome-headless-shell-linux64/chrome-headless-shell"
if [ ! -x "$SHELL_BIN" ]; then
    echo "[serve] FATAL: playwright wants chromium rev $EXPECT, not found at" >&2
    echo "        $SHELL_BIN" >&2
    echo "        Shared disk currently has:" >&2
    ls -d "$PLAYWRIGHT_BROWSERS_PATH"/*/ 2>/dev/null | xargs -n1 basename >&2 || true
    echo "" >&2
    echo "        Do NOT run 'playwright install' on this path — it deletes the" >&2
    echo "        existing revision first and then dies on __dirlock under a shared network filesystem," >&2
    echo "        which is how a working revision can be lost. Recovery:" >&2
    echo "          curl -fL -o /tmp/shell.zip https://cdn.playwright.dev/builds/cft/<ver>/linux64/chrome-headless-shell-linux64.zip" >&2
    echo "          unzip -q /tmp/shell.zip -d \$PLAYWRIGHT_BROWSERS_PATH/chromium_headless_shell-${EXPECT}" >&2
    echo "          touch \$PLAYWRIGHT_BROWSERS_PATH/chromium_headless_shell-${EXPECT}/INSTALLATION_COMPLETE" >&2
    echo "        (the <ver> build string is printed by the failing playwright install)" >&2
    exit 1
fi
echo "[serve] chromium rev $EXPECT OK"

python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
    print('[serve] chromium launches:', b.version); b.close()
"

cd "$GRADER"
echo "[serve] code: $(git log --oneline -1)"
PORT=80
# `"$@"` is forwarded verbatim below, so anything this script does not reject
# here reaches argparse. --rubric used to be sniffed out to name the dump file;
# the service has one reward now and no --rubric flag, so passing it would exit
# on "unrecognized arguments" — say that instead of leaving it to argparse.
for _a in "$@"; do
  case "${_a}" in
    --port|--port=*)     echo "[serve] --port is set by this script (--port ${PORT})" >&2; exit 1 ;;
    --rubric|--rubric=*) echo "[serve] --rubric is gone: the service serves the webdev group reward only" >&2; exit 1 ;;
  esac
done

DUMP_DEFAULT=${DESIGN_GRADER_ROOT}/dump/grades-group.jsonl
mkdir -p "$(dirname "${DUMP_DEFAULT}")"
HAS_DUMP=0
for _a in "$@"; do case "${_a}" in --dump|--dump=*) HAS_DUMP=1 ;; esac; done
DUMP_ARGS=()
if [ "$HAS_DUMP" = 0 ]; then DUMP_ARGS=(--dump "${DUMP_DEFAULT}"); fi
echo "[serve] dump=$([ "$HAS_DUMP" = 1 ] && echo '(from args)' || echo "${DUMP_DEFAULT}")"

exec env PYTHONPATH=src python3 -m design_grader.service.server --port "${PORT}" "${DUMP_ARGS[@]}" "$@"
