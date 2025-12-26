import express from "express";
import bodyParser from "body-parser";
import fs from "fs";
import path from "path";
import { chromium } from "playwright";

const app = express();
app.use(bodyParser.json({ limit: "1mb" }));

const PORT = process.env.PORT || 7003;
const OUTPUT_DIR = process.env.OUTPUT_DIR || "/home/witness/memory/webresearch";
const HEADLESS = process.env.HEADLESS !== "0";
const MONITOR_LOG_DIR = process.env.MONITOR_LOG_DIR || "/data/_logs";
const MONITOR_LOG_FILE = path.join(MONITOR_LOG_DIR, "mcp-playwright.jsonl");

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(MONITOR_LOG_DIR, { recursive: true });

function monitorLog(category, message, payload = {}) {
  const record = {
    ts: new Date().toISOString(),
    module: "mcp-playwright",
    category,
    message,
    payload,
  };
  try {
    fs.appendFileSync(MONITOR_LOG_FILE, `${JSON.stringify(record)}\n`, "utf-8");
  } catch (e) {
    // best-effort logging only
  }
}

function slugify(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]+/gi, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 50) || "item";
}

function normalizeTimeout(raw) {
  const ms = Number(raw) || 0;
  if (!ms) return 20_000;
  return Math.max(5_000, Math.min(ms, 60_000));
}

async function collectArticle(context, url, timeoutMs) {
  const page = await context.newPage();
  try {
    await page.goto(url, {
      waitUntil: "networkidle",
      timeout: normalizeTimeout(timeoutMs),
    });
    const title = (await page.title()) || url;
    const resolvedUrl = page.url();
    const bodyText =
      (await page.innerText("main").catch(() => null)) ||
      (await page.innerText("article").catch(() => null)) ||
      (await page.innerText("body").catch(() => null)) ||
      "";
    const htmlSnapshot = await page.content();
    const body = [
      (bodyText || "").trim(),
      "",
      "----- FULL HTML SNAPSHOT -----",
      htmlSnapshot || "",
    ]
      .join("\n")
      .trim();
    return {
      title: title.trim(),
      url: resolvedUrl,
      body,
      plainText: (bodyText || "").trim(),
    };
  } finally {
    await page.close();
  }
}

async function crawlKeyword(context, keyword, perKeyword) {
  const page = await context.newPage();
  try {
    const searchUrl = `https://news.google.com/search?q=${encodeURIComponent(
      keyword
    )}&hl=ko&gl=KR&ceid=KR:ko`;
    await page.goto(searchUrl, { waitUntil: "domcontentloaded", timeout: 45_000 });
    const links = await page.$$eval("article h3 a", (anchors) =>
      anchors.slice(0, 12).map((a) => ({
        href: a.href,
        title: a.innerText || "",
      }))
    );

    const normalized = [];
    for (const item of links) {
      try {
        const absoluteUrl = new URL(item.href, "https://news.google.com").toString();
        normalized.push({
          url: absoluteUrl,
          title: item.title?.trim() || "",
        });
      } catch (_) {
        // skip malformed URL
      }
      if (normalized.length >= perKeyword) break;
    }
    return normalized;
  } finally {
    await page.close();
  }
}

app.post("/crawl", async (req, res) => {
  const rawKeywords = Array.isArray(req.body?.keywords) ? req.body.keywords : [];
  const keywords = rawKeywords
    .map((k) => String(k || "").trim())
    .filter(Boolean)
    .slice(0, 5);
  const perKeyword = Math.max(1, Math.min(Number(req.body?.perKeyword) || 2, 5));

  if (!keywords.length) {
    return res.status(400).json({ success: false, error: "keywords required" });
  }

  monitorLog("api_call", "crawl start", { keywords, perKeyword });

  const savedFiles = [];
  const articles = []; // aggregated results to return
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({
    userAgent:
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
  });

  try {
    for (const keyword of keywords) {
      // 검색 결과 목록은 별도 변수로 받아서 아래의 누적 배열과 구분
      const hits = await crawlKeyword(context, keyword, perKeyword);
      let idx = 1;
      for (const article of hits) {
        const detail = await collectArticle(context, article.url);
        const now = new Date();
        const tsDate = now.toISOString().slice(2, 10).replace(/-/g, ""); // YYMMDD
        const tsTime = now.toISOString().slice(11, 16).replace(":", ""); // HHMM
        // idx를 붙여 같은 분에 여러 기사 저장 시 덮어쓰지 않도록 방지
        const filename = `${tsDate}-${tsTime}_${slugify(keyword)}_${idx}.txt`;
        const content =
          `keyword: ${keyword}\n` +
          `source_url: ${detail.url}\n` +
          `title: ${detail.title.replace(/\n/g, " ")}\n` +
          `saved_at: ${now.toISOString()}\n\n` +
          (detail.body || "(no content)");
        fs.writeFileSync(path.join(OUTPUT_DIR, filename), content, "utf8");
        savedFiles.push(path.join(OUTPUT_DIR, filename));
        const summarySource = detail.plainText || detail.body || "";
        const summary = summarySource
          .replace(/\s+/g, " ")
          .trim()
          .slice(0, 400);
        // 응답으로 반환할 기사 목록 누적
        articles.push({
          keyword,
          url: detail.url,
          title: detail.title,
          summary,
          file: path.join(OUTPUT_DIR, filename),
        });
        idx += 1;
      }
    }
    res.json({
      success: true,
      saved_files: savedFiles,
      count: savedFiles.length,
      keywords,
      articles,
    });
    monitorLog("api_call", "crawl ok", {
      keywords,
      perKeyword,
      saved_count: savedFiles.length,
    });
  } catch (err) {
    monitorLog("api_call", "crawl failed", { error: String(err) });
    res.status(500).json({ success: false, error: String(err) });
  } finally {
    await context.close();
    await browser.close();
  }
});

app.post("/fetch", async (req, res) => {
  const url = String(req.body?.url || req.body?.link || "").trim();
  const timeoutMs = req.body?.timeout_ms ?? req.body?.timeoutMs;

  if (!url) {
    return res.status(400).json({ success: false, error: "url required" });
  }

  monitorLog("api_call", "fetch start", { url });

  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({
    userAgent:
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
  });

  try {
    const detail = await collectArticle(context, url, timeoutMs);
    return res.json({
      success: true,
      data: {
        link: detail.url,
        title: detail.title,
        body: detail.body,
        plain_text: detail.plainText,
      },
    });
    monitorLog("api_call", "fetch ok", { url: detail.url });
  } catch (err) {
    monitorLog("api_call", "fetch failed", { url, error: String(err) });
    return res
      .status(500)
      .json({ success: false, error: String(err || "fetch_failed") });
  } finally {
    await context.close();
    await browser.close();
  }
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, output_dir: OUTPUT_DIR });
});

app.listen(PORT, () => {
  console.log(`Playwright MCP listening on ${PORT}`);
});
