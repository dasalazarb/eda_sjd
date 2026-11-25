const assert = require("node:assert/strict");
const fs = require("node:fs");
const {chromium} = require("playwright");

async function main() {
  const browser = await chromium.launch({
    executablePath: process.argv[3],
    args: ["--no-sandbox", "--disable-breakpad", "--disable-crash-reporter", "--no-zygote"],
    timeout: 10000,
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.setContent(fs.readFileSync(process.argv[2], "utf8"), {waitUntil: "domcontentloaded"});
    const variables = page.locator("#variableTable");
    assert.match(await variables.getByRole("status").textContent(), /Page 1 of \d+/);
    await variables.getByRole("button", {name: "Next"}).click();
    assert.match(await variables.getByRole("status").textContent(), /Page 2 of/);
    await page.locator("#search").fill("1300");
    assert.match(await variables.getByRole("status").textContent(), /2 rows/);
    await page.locator("#namespace").selectOption("ext");
    assert.equal(await variables.getByRole("status").textContent(), "No rows");
    await page.locator("#namespace").selectOption("all");
    await page.locator("#kind").selectOption("ana");
    assert.match(await page.locator("#specialTable [role=status]").textContent(), /Page 1 of/);

    assert.equal(await page.locator("img").count(), 0);
    const exported = await page.evaluate(async () => {
      let blob;
      const oldCreate = URL.createObjectURL;
      const oldClick = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = value => { blob = value; return "blob:report-test"; };
      HTMLAnchorElement.prototype.click = function () {};
      try {
        const payload = JSON.parse(document.getElementById("payload").textContent);
        document.querySelector("[data-export='all-json']").click();
        const all = JSON.parse(await blob.text());
        document.getElementById("which").value = "lab__other__text";
        document.querySelector("[data-export='variable-csv']").click();
        const csv = await blob.text();
        return [payload.details.length, all.length, csv, csvCell(" \t=1+1")];
      } finally {
        URL.createObjectURL = oldCreate;
        HTMLAnchorElement.prototype.click = oldClick;
      }
    });
    assert.equal(exported[0], exported[1]);
    assert.match(exported[2], /<\/script><img/);
    assert.match(exported[3], /^"'/);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
