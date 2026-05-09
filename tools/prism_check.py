"""Quick check: are we logged in? What does Prism show with current session?"""
import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

SESSION_FILE = Path("prism_session.json")
PROJECT_URL  = "https://prism.openai.com/?u=f2289688-d3b2-47dd-b269-f06eb24f6f0d&pg=1&m=main.tex"

async def check():
    async with async_playwright() as p:
        storage = json.loads(SESSION_FILE.read_text())
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(storage_state=storage)
        page    = await ctx.new_page()
        await page.goto(PROJECT_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(6000)
        print("URL:  ", page.url)
        print("Title:", await page.title())
        body = await page.inner_text("body")
        print("Body (first 400):", body[:400])
        await page.screenshot(path="output/prism_check.png")
        print("Screenshot saved to output/prism_check.png")
        await browser.close()

asyncio.run(check())
