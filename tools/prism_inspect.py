"""List all file names in the Prism project from the sidebar file tree."""
import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

SESSION_FILE = Path("prism_session.json")
PROJECT_URL  = "https://prism.openai.com/?u=f2289688-d3b2-47dd-b269-f06eb24f6f0d&pg=1&m=main.tex"

async def inspect():
    async with async_playwright() as p:
        storage = json.loads(SESSION_FILE.read_text())
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(storage_state=storage)
        page    = await ctx.new_page()
        await page.goto(PROJECT_URL, wait_until="networkidle", timeout=60000)
        await page.locator(".monaco-editor").first.wait_for(timeout=30000)
        await page.wait_for_timeout(6000)

        # Try to get file names from the DOM (file tree / tabs)
        file_names = await page.evaluate("""
            () => {
                // Try tab labels first
                const tabs = Array.from(document.querySelectorAll('[class*="tab"] [class*="label"], [class*="tab-label"], .tab .label'))
                    .map(el => el.textContent.trim()).filter(Boolean);
                // Try file tree items
                const tree = Array.from(document.querySelectorAll('[class*="file-label"], [class*="filename"], [class*="tree-item"] span, [class*="explorer"] span'))
                    .map(el => el.textContent.trim()).filter(t => t.includes('.'));
                // Try monaco model URIs for any non-inmemory URIs
                const uris = window.monaco?.editor?.getModels()?.map(m => m.uri.toString()) || [];
                return { tabs, tree, uris };
            }
        """)
        print("Tabs:", file_names.get("tabs"))
        print("Tree:", file_names.get("tree"))
        print("URIs:", file_names.get("uris"))

        # Also grab all text content that looks like filenames
        all_text = await page.evaluate("""
            () => Array.from(document.querySelectorAll('*'))
                .map(el => el.childNodes)
                .reduce((a, b) => [...a, ...b], [])
                .filter(n => n.nodeType === 3)
                .map(n => n.textContent.trim())
                .filter(t => /\.(tex|bib|md|txt|cls|sty)$/.test(t))
        """)
        print("Filename-like text nodes:", list(set(all_text)))

        await browser.close()

asyncio.run(inspect())
