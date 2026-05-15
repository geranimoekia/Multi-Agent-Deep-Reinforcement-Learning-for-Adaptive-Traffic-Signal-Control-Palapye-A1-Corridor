"""Push docs/main.tex and docs/references.bib to Prism."""
import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

SESSION_FILE = Path("prism_session.json")
BASE_URL     = "https://prism.openai.com/?u=f2289688-d3b2-47dd-b269-f06eb24f6f0d&pg=1"

FILES = [
    ("docs/main.tex",       "main.tex"),
    ("docs/references.bib", "references.bib"),
]

async def push_file(page, local_path: str, prism_filename: str):
    content = Path(local_path).read_text(encoding="utf-8")
    url = f"{BASE_URL}&m={prism_filename}"
    print(f"  → {prism_filename} ...", end=" ", flush=True)
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.locator(".monaco-editor").first.wait_for(timeout=30000)
    await page.wait_for_timeout(8000)

    await page.evaluate("""
        ({ content }) => {
            const editors = window.monaco?.editor?.getEditors() || [];
            if (!editors.length) return false;
            const model = editors[0].getModel();
            if (!model) return false;
            model.setValue(content);
            editors[0].focus();
            return true;
        }
    """, {"content": content})

    await page.keyboard.press("Control+s")
    await page.wait_for_timeout(2000)
    await page.keyboard.press("Control+s")
    await page.wait_for_timeout(6000)

    info = await page.evaluate("""
        () => {
            const editors = window.monaco?.editor?.getEditors() || [];
            const m = editors.length ? editors[0].getModel() : null;
            return m ? {lines: m.getLineCount(), chars: m.getValueLength()} : {lines: -1, chars: -1};
        }
    """)
    local_lines = len(content.splitlines())
    local_chars = len(content)
    if info["lines"] == local_lines and info["chars"] == local_chars:
        print(f"✓  ({local_lines} lines / {local_chars} chars)")
    else:
        print(f"✗  mismatch — local {local_lines}L/{local_chars}c | remote {info['lines']}L/{info['chars']}c")

async def push():
    async with async_playwright() as p:
        storage = json.loads(SESSION_FILE.read_text())
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(storage_state=storage)
        page    = await ctx.new_page()
        print("Pushing to Prism...")
        for local_path, prism_filename in FILES:
            await push_file(page, local_path, prism_filename)
        await browser.close()
        print("\nDone.")

asyncio.run(push())
