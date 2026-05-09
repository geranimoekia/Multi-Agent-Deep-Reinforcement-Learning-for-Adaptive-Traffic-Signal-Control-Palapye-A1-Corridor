import json
from pathlib import Path

s = json.loads(Path("prism_session.json").read_text())
print(f"Total cookies: {len(s['cookies'])}\n")
for c in s["cookies"]:
    val = c["value"][:40] + "..." if len(c["value"]) > 40 else c["value"]
    empty = " <EMPTY>" if not c["value"] else ""
    print(f"  {c['domain']:35s}  {c['name']:35s}  {val}{empty}")
