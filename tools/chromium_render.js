// Render an SVG to a transparent PNG with Chromium (Skia), as an independent
// rasteriser to check our own against.
//   node tools/chromium_render.js in.svg out.png 2048 2048
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [svgPath, outPath, w, h] = process.argv.slice(2);
  const W = parseInt(w, 10), H = parseInt(h, 10);
  let svg = fs.readFileSync(svgPath, 'utf8');
  svg = svg.replace(/<svg([^>]*?)\swidth="[^"]*"\sheight="[^"]*"/, `<svg$1 width="${W}" height="${H}"`);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  await page.setContent(`<!doctype html><html><body style="margin:0;background:transparent">${svg}</body></html>`);
  await page.screenshot({ path: outPath, omitBackground: true, clip: { x: 0, y: 0, width: W, height: H } });
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
