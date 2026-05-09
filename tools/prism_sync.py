"""
prism_sync.py — Auto-sync a local .tex file to prism.openai.com

SETUP (run once):
    pip install playwright watchdog
    playwright install chromium

FIRST RUN:
    python prism_sync.py --setup
    A browser window opens. Log in to Prism manually, open your project,
    then press Enter in the terminal. Session is saved to prism_session.json
    and reused on every future run — you won't need to log in again.

NORMAL USE:
    python prism_sync.py --file "my_paper.tex" --project-url "https://prism.openai.com/project/YOUR_ID"

    Watches the file for saves. Every time you save, the updated content
    is pushed into the open Prism editor automatically.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from playwright.async_api import async_playwright, BrowserContext

PRISM_URL    = "https://prism.openai.com"
SESSION_FILE = Path("prism_session.json")

# ── Playwright helpers ────────────────────────────────────────────────────────

async def save_session(context: BrowserContext):
    storage = await context.storage_state()
    SESSION_FILE.write_text(json.dumps(storage, indent=2))
    print(f"[PRISM] Session saved to {SESSION_FILE}")


async def load_context(playwright, headless: bool = True):
    browser = await playwright.chromium.launch(headless=headless)
    if SESSION_FILE.exists():
        storage = json.loads(SESSION_FILE.read_text())
        context = await browser.new_context(storage_state=storage)
        print("[PRISM] Loaded saved session")
    else:
        context = await browser.new_context()
        print("[PRISM] No saved session — starting fresh")
    return browser, context


# ── Setup: one-time login ─────────────────────────────────────────────────────

async def setup():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()
        await page.goto(PRISM_URL)
        print("\n[PRISM] Browser opened.")
        print("  1. Log in to your OpenAI account")
        print("  2. Open your Prism project")
        print("  3. Press Enter here when you're ready\n")
        input("Press Enter to save session...")
        await save_session(context)
        await browser.close()
    print("[PRISM] Setup complete. Run push_once.py to sync.")


# ── Sync: push file content into the Prism editor ────────────────────────────

async def push_to_prism(context: BrowserContext, project_url: str, content: str):
    page = await context.new_page()
    try:
        print(f"[PRISM] Opening project...")
        await page.goto(project_url, wait_until="domcontentloaded", timeout=30_000)
        await page.locator(".monaco-editor").first.wait_for(timeout=30_000)
        await page.wait_for_timeout(8000)

        # Target the ACTIVE editor model — this is the file shown by the URL
        # (not the largest/smallest model, which may be a different .tex file)
        replaced = await page.evaluate(
            """
            ({ content }) => {
                const editors = window.monaco?.editor?.getEditors() || [];
                if (!editors.length) return false;
                const model = editors[0].getModel();
                if (!model) return false;
                model.setValue(content);
                editors[0].focus();
                return true;
            }
            """,
            {"content": content},
        )

        if not replaced:
            raise RuntimeError("Could not find active Monaco editor model")

        # Trigger save so Prism persists the change to its server
        await page.keyboard.press("Control+s")
        await page.wait_for_timeout(3000)
        print(f"[PRISM] Content pushed ({len(content)} chars)")

    except Exception as e:
        print(f"[PRISM ERROR] Could not push: {e}")
        print("  → Check that project_url is correct and you are logged in.")
        print("  → Re-run with --setup if session has expired.")
    finally:
        await page.close()


# ── File watcher ─────────────────────────────────────────────────────────────

class TexHandler(FileSystemEventHandler):
    def __init__(self, tex_path: Path, loop: asyncio.AbstractEventLoop,
                 context: BrowserContext, project_url: str):
        self.tex_path    = tex_path.resolve()
        self.loop        = loop
        self.context     = context
        self.project_url = project_url
        self._pending    = False   # debounce: skip duplicate events in quick succession

    def on_modified(self, event):
        if Path(event.src_path).resolve() != self.tex_path:
            return
        if self._pending:
            return
        self._pending = True
        self.loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._sync()
        )

    async def _sync(self):
        await asyncio.sleep(0.5)   # short debounce — wait for editor to finish writing
        self._pending = False
        try:
            content = self.tex_path.read_text(encoding="utf-8")
            print(f"\n[WATCH] {self.tex_path.name} changed — syncing to Prism...")
            await push_to_prism(self.context, self.project_url, content)
        except Exception as e:
            print(f"[WATCH ERROR] {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def watch(tex_file: str, project_url: str):
    tex_path = Path(tex_file)
    if not tex_path.exists():
        print(f"[ERROR] File not found: {tex_path}")
        sys.exit(1)
    if not SESSION_FILE.exists():
        print("[ERROR] No saved session. Run:  python prism_sync.py --setup")
        sys.exit(1)

    async with async_playwright() as p:
        browser, context = await load_context(p, headless=True)

        loop    = asyncio.get_event_loop()
        handler = TexHandler(tex_path, loop, context, project_url)
        observer = Observer()
        observer.schedule(handler, str(tex_path.parent), recursive=False)
        observer.start()

        print(f"\n[PRISM SYNC] Watching: {tex_path.resolve()}")
        print(f"[PRISM SYNC] Target:   {project_url}")
        print(f"[PRISM SYNC] Save your .tex file to trigger a sync. Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[PRISM SYNC] Stopped.")
        finally:
            observer.stop()
            observer.join()
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Auto-sync .tex → prism.openai.com")
    parser.add_argument("--setup", action="store_true",
                        help="Log in and save session (run once)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to your .tex file to watch")
    parser.add_argument("--project-url", type=str, default=None,
                        help="Full URL of your Prism project")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(setup())
    elif args.file and args.project_url:
        asyncio.run(watch(args.file, args.project_url))
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python prism_sync.py --setup')
        print('  python prism_sync.py --file "thesis.tex" --project-url "https://prism.openai.com/project/abc123"')


if __name__ == "__main__":
    main()
