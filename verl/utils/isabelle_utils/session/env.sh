# Source this to enable Isabelle 2025 on any datatech node:
#   source /2022533109/zhouchuyan/isabelle/env.sh
# ISABELLE_HOME_USER points DIRECTLY at the shared user dir (heaps / cache /
# etc / servers.db). All nodes share it with NO dependency on a node-local
# ~/.isabelle symlink. That symlink lived in the ephemeral container root and
# was wiped on every container/tmux-server restart -> missing prebuilt heaps
# -> session_start failures -> training hang at the first step (2026-07-07
# incident). Setting the env var here is persistent and self-healing.
export ISABELLE_HOME="/2022533109/zhouchuyan/isabelle/Isabelle2025"
export ISABELLE_HOME_USER="/2022533109/zhouchuyan/isabelle/user"
case ":$PATH:" in
  *":$ISABELLE_HOME/bin:"*) ;;
  *) export PATH="$ISABELLE_HOME/bin:$PATH" ;;
esac
