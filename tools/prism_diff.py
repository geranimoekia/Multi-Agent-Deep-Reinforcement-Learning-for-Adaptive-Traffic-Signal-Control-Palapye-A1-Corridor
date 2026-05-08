"""
prism_diff.py - Fetch content from Prism and diff against local file.
Usage:
    python tools/prism_diff.py --file "docs/main.tex" --project-url "URL"
"""
import argparse
import asyncio
import difflib
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import async_playwright

SESSION_FILE = Path("prism_session.json")
PROFILE_DIR  = Path("prism_profile")


async def fetch_from_prism(project_url: str) -> str:
    async with async_playwright() as p:
        if PROFILE_DIR.exists():
            context = await p.chromium.launch_persistent_context(
                str(PROFILE_DIR), headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            browser = None
        else:
            browser = await p.chromium.launch(headless=True)
            storage = json.loads(SESSION_FILE.read_text())
            context = await browser.new_context(storage_state=storage)

        page = await context.new_page()
        print("[PRISM] Fetching content from Prism...")
        await page.goto(project_url, wait_until="domcontentloaded", timeout=30_000)
        await page.locator(".monaco-editor").first.wait_for(timeout=30_000)

        file_name = parse_qs(urlparse(project_url).query).get("m", [""])[0]
        content = await page.evaluate(
            """
            ({ fileName }) => {
                if (!window.monaco?.editor) return null;
                const models = window.monaco.editor
                    .getModels()
                    .filter((m) => m.getLanguageId() === "latex");
                if (!models.length) return null;
                const target = fileName === "main.tex"
                    ? models.reduce((best, m) =>
                        m.getValueLength() < best.getValueLength() ? m : best)
                    : models.find((m) => m.getValue().includes("\\\\documentclass")) || models[0];
                return target.getValue();
            }
            """,
            {"fileName": file_name},
        )

        await page.close()
        if browser:
            await browser.close()
        else:
            await context.close()

        if content is None:
            print("[ERROR] Could not read Monaco editor content.")
            sys.exit(1)

        return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--project-url", required=True)
    args = parser.parse_args()

    local_path = Path(args.file)
    if not local_path.exists():
        print(f"[ERROR] Local file not found: {local_path}")
        sys.exit(1)

    if not SESSION_FILE.exists() and not PROFILE_DIR.exists():
        print("[ERROR] No saved session. Run: python tools/prism_sync.py --setup")
        sys.exit(1)

    local_text  = local_path.read_text(encoding="utf-8")
    remote_text = asyncio.run(fetch_from_prism(args.project_url))

    local_lines  = local_text.splitlines(keepends=True)
    remote_lines = remote_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        remote_lines, local_lines,
        fromfile="prism (remote)",
        tofile=f"{local_path} (local)",
        lineterm=""
    ))

    if not diff:
        print("\nFiles are IDENTICAL. Local and Prism are in sync.")
    else:
        added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        print(f"\nDiff: +{added} lines in local  /  -{removed} lines in Prism\n")
        print("".join(diff[:200]))
        if len(diff) > 200:
            print(f"\n... ({len(diff)-200} more diff lines truncated)")


if __name__ == "__main__":
    main()
