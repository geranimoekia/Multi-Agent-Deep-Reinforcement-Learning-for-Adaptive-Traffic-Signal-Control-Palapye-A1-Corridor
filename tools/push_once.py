"""Push docs/main.tex to the active Monaco model in Prism (= the file shown by the m=main.tex URL)."""
import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

SESSION_FILE = Path("prism_session.json")
PROJECT_URL  = "https://prism.openai.com/?u=f2289688-d3b2-47dd-b269-f06eb24f6f0d&pg=1&m=main.tex"
content = Path("docs/main.tex").read_text(encoding="utf-8")

async def push():
    async with async_playwright() as p:
        storage = json.loads(SESSION_FILE.read_text())
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(storage_state=storage)
        page    = await ctx.new_page()
        print("[PUSH] Navigating...")
        await page.goto(PROJECT_URL, wait_until="networkidle", timeout=60000)
        await page.locator(".monaco-editor").first.wait_for(timeout=30000)
        await page.wait_for_timeout(8000)

        info_before = await page.evaluate("""
            () => {
                const editors = window.monaco?.editor?.getEditors() || [];
                const m = editors.length ? editors[0].getModel() : null;
                return m ? {chars: m.getValueLength(), lines: m.getLineCount()} : {chars: -1, lines: -1};
            }
        """)
        print(f"[BEFORE] Active model: {info_before['lines']} lines / {info_before['chars']} chars")

        # Update the ACTIVE editor's model (= the file Prism is showing for m=main.tex)
        replaced = await page.evaluate("""
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
        print(f"[PUSH]  setValue: {replaced}  ({len(content)} chars)")

        # Ctrl+S to trigger Prism's save
        await page.keyboard.press("Control+s")
        await page.wait_for_timeout(2000)
        await page.keyboard.press("Control+s")
        print("[PUSH]  Ctrl+S sent — waiting for server save...")
        await page.wait_for_timeout(6000)

        info_after = await page.evaluate("""
            () => {
                const editors = window.monaco?.editor?.getEditors() || [];
                const m = editors.length ? editors[0].getModel() : null;
                if (!m) return {lines: -1, yolo: false, sumo: false, marl: false};
                const v = m.getValue();
                return {
                    lines: m.getLineCount(),
                    yolo: v.includes("You Only Look Once"),
                    sumo: v.includes("Traffic Simulation Tools"),
                    marl: v.includes("Dec-POMDP"),
                };
            }
        """)
        print(f"\n[VERIFY] Remote lines: {info_after['lines']}  (local: {len(content.splitlines())})")
        print(f"[VERIFY] Has YOLO section:          {info_after['yolo']}")
        print(f"[VERIFY] Has SUMO tools section:    {info_after['sumo']}")
        print(f"[VERIFY] Has MARL general intro:    {info_after['marl']}")
        await browser.close()

asyncio.run(push())
