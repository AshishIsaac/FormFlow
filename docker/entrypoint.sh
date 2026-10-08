#!/usr/bin/env sh
# First start: download the pose models into the /workspace volume (once). FORMFLOW_SKIP_SETUP=1 skips it
# (CI, or when the models are mounted in).
set -e
if [ -z "$FORMFLOW_SKIP_SETUP" ]; then
  formflow models download || echo "model download failed - the app will retry when it starts"
fi
exec formflow "$@"
