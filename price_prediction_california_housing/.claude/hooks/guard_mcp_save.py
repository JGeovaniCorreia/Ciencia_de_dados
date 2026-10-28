"""PreToolUse hook: only subagents may create records in the ml-context MCP.

Claude Code adds ``agent_type`` to the hook input only when the tool call comes
from a subagent. Calls from the main session lack it and are blocked (exit 2),
with the reason fed back to the model. Updates and deletes of existing records
are not covered here: they remain main-session actions on explicit owner request.
"""

import json
import sys


def main() -> int:
    """Read the hook payload from stdin and decide whether to allow the call.

    Returns:
        0 to allow the tool call, 2 to block it.
    """
    try:
        payload: dict = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0  # Malformed input: do not break the session over the guard itself.

    if payload.get("agent_type"):
        return 0

    sys.stderr.reconfigure(encoding="utf-8")  # Windows console default is cp1252.
    print(
        "Bloqueado: no projeto, só os subagentes gravam conhecimento novo no MCP "
        "(save_knowledge). Delegue a gravação ao agente responsável pela decisão.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
