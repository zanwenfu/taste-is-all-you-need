// Renders banner.html in Chrome: every frame of the moving banner, or one still.
//   node render.cjs frames <variant> <out-dir> <frames> <fps>
//   node render.cjs still <variant> <file>
// The banner is 1280 x 400 drawn at 2x; the social card (variant og) is 1280 x 640 at 1x.
const { chromium } = require("@playwright/test");
const [MODE, VARIANT, OUT, FRAMES, FPS] = process.argv.slice(2);
(async () => {
  const og = VARIANT === "og";
  const browser = await chromium.launch({ channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1280, height: og ? 640 : 400 }, deviceScaleFactor: og ? 1 : 2 });
  page.on("pageerror", (e) => { console.error("page error:", e.message); process.exit(1); });
  await page.goto(`file://${__dirname}/banner.html?variant=${VARIANT}`);
  await page.waitForFunction(() => window.ready === true);
  if (MODE === "still") {
    await page.evaluate(() => window.draw(null));
    await page.screenshot({ path: OUT });
  } else {
    for (let i = 0; i < +FRAMES; i++) {
      await page.evaluate((t) => window.draw(t), i / +FPS);
      await page.screenshot({ path: `${OUT}/f_${String(i).padStart(4, "0")}.png` });
    }
  }
  await browser.close();
})().catch((e) => { console.error("render failed:", e.message); process.exit(1); });
