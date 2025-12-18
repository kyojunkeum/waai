import express from "express";
import fs from "fs";
import path from "path";
import url from "url";
import YAML from "yaml";
import os from "os";

const app = express();
const port = process.env.PORT || 7001;

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

function resolveDataPath(envVar, defaultSubpath) {
  const candidates = [];
  const envValue = process.env[envVar];
  if (envValue) {
    candidates.push(envValue);
  }
  candidates.push(path.join("/data", defaultSubpath));
  candidates.push(path.join(REPO_ROOT, "data", defaultSubpath));
  candidates.push(path.join(os.homedir(), ".waai", defaultSubpath));

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[candidates.length - 1];
}

const DIARY_ROOT = resolveDataPath("DIARY_ROOT", "diary");
const IDEAS_ROOT = resolveDataPath("IDEAS_ROOT", "ideas");
const WEB_RESEARCH_ROOT = resolveDataPath("WEB_RESEARCH_ROOT", "web_research");
const WORKS_ROOT = resolveDataPath("WORKS_ROOT", "works");
const BIBLE_ROOT = resolveDataPath("BIBLE_ROOT", "bible");
const DEFAULT_LIMIT_PER_TYPE = parseInt(process.env.LIMIT_PER_TYPE || "10", 10);
const DEFAULT_PREVIEW_CHARS = parseInt(process.env.PREVIEW_CHARS || "1200", 10);

app.use(express.json());

function listFilesInRoot(root) {
  const results = [];
  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.isFile()) results.push(full);
    }
  }
  if (fs.existsSync(root)) {
    walk(root);
  }
  return results;
}

function listFilesInRootWithMtime(root) {
  const files = listFilesInRoot(root);
  return files
    .map((p) => ({
      abs: p,
      rel: path.relative(root, p),
      mtime: fs.statSync(p).mtimeMs,
    }))
    .sort((a, b) => b.mtime - a.mtime); // 최신순
}

function listFiles() {
  return listFilesInRoot(DIARY_ROOT);
}

function listCategorized() {
  return {
    diary: listFilesInRoot(DIARY_ROOT).map((p) => path.relative(DIARY_ROOT, p)),
    ideas: listFilesInRoot(IDEAS_ROOT).map((p) => path.relative(IDEAS_ROOT, p)),
    web_research: listFilesInRoot(WEB_RESEARCH_ROOT).map((p) =>
      path.relative(WEB_RESEARCH_ROOT, p)
    ),
    works: listFilesInRoot(WORKS_ROOT).map((p) => path.relative(WORKS_ROOT, p)),
    bible: listFilesInRoot(BIBLE_ROOT).map((p) => path.relative(BIBLE_ROOT, p)),
  };
}

function safeReadFile(root, name) {
  const rootPath = path.resolve(root);
  const fullPath = path.resolve(rootPath, name);
  const relative = path.relative(rootPath, fullPath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("invalid path");
  }
  if (!fs.existsSync(fullPath)) {
    throw new Error("file not found");
  }
  return {
    name,
    content: fs.readFileSync(fullPath, "utf-8"),
  };
}

function parseFrontMatter(text) {
  const trimmed = text.trimStart();
  if (!trimmed.startsWith("---")) return { meta: {}, body: text };

  const parts = trimmed.split("---");
  if (parts.length < 3) return { meta: {}, body: text };

  const metaRaw = parts[1];
  const body = parts.slice(2).join("---");

  let meta = {};
  try {
    meta = YAML.parse(metaRaw) || {};
  } catch (e) {
    meta = {};
  }
  return { meta, body };
}

function parseDateField(val) {
  if (!val) return null;
  if (val instanceof Date) return val;
  if (typeof val === "string") {
    const d = Date.parse(val);
    if (!Number.isNaN(d)) return new Date(d);
  }
  return null;
}

function parseDateFromFilename(relPath) {
  const name = path.basename(relPath, path.extname(relPath));
  const cand = name.slice(0, 10);
  const d = Date.parse(cand);
  if (!Number.isNaN(d)) return new Date(d);
  return null;
}

function asList(val) {
  if (val === undefined || val === null) return [];
  if (Array.isArray(val)) return val.map((v) => String(v));
  return [String(val)];
}

function matchesFilter(meta, body, relPath, startDate, endDate, keyword) {
  const metaDate =
    parseDateField(meta.date) ||
    parseDateField(meta.created_at) ||
    parseDateField(meta.updated_at) ||
    parseDateFromFilename(relPath);

  const start = startDate ? parseDateField(startDate) : null;
  const end = endDate ? parseDateField(endDate) : null;

  if (start && metaDate && metaDate < start) return false;
  if (end && metaDate && metaDate > end) return false;

  if (keyword) {
    const fields = [
      meta.title || "",
      ...asList(meta.tags),
      ...asList(meta.topics),
      ...asList(meta.people),
      ...asList(meta.locations),
      body || "",
    ];
    if (!fields.some((v) => v && v.includes(keyword))) {
      return false;
    }
  }
  return true;
}

function buildPreview(body, previewChars) {
  if (!body) return "";
  const text = body.trim();
  if (text.length <= previewChars) return text;
  return `${text.slice(0, previewChars)}\n...[본문 일부만 포함]`;
}

app.get("/files", (req, res) => {
  const files = listFiles().map((p) => path.relative(DIARY_ROOT, p));
  res.json({ files });
});

app.get("/file", (req, res) => {
  const name = req.query.name;
  if (!name) return res.status(400).json({ error: "name query required" });
  try {
    const { content } = safeReadFile(DIARY_ROOT, name);
    res.json({ name, content });
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }
});

// 새: 카테고리별 파일 목록
app.get("/files-all", (req, res) => {
  res.json(listCategorized());
});

// 새: 카테고리 지정 파일 읽기
app.get("/file-by-type", (req, res) => {
  const { type, name } = req.query;
  if (!type || !name) {
    return res.status(400).json({ error: "type and name query required" });
  }
  const roots = {
    diary: DIARY_ROOT,
    ideas: IDEAS_ROOT,
    web_research: WEB_RESEARCH_ROOT,
    works: WORKS_ROOT,
    bible: BIBLE_ROOT,
  };
  const root = roots[type];
  if (!root) return res.status(400).json({ error: "unknown type" });
  try {
    const { content } = safeReadFile(root, name);
    res.json({ type, name, content });
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }
});

app.get("/search", (req, res) => {
  const keyword = req.query.keyword;
  if (!keyword) return res.status(400).json({ error: "keyword required" });
  const matches = [];
  for (const p of listFiles()) {
    const text = fs.readFileSync(p, "utf-8");
    if (text.includes(keyword)) {
      matches.push({
        name: path.relative(DIARY_ROOT, p),
        snippet: text.slice(0, 200)
      });
    }
  }
  res.json({ keyword, matches });
});

// 새: 필터 + 메타/미리보기 반환 (다중 타입)
app.post("/filter", (req, res) => {
  const {
    include,
    start_date,
    end_date,
    keyword,
    limit_per_type,
    preview_chars,
  } = req.body || {};

  const types = (include && Array.isArray(include)
    ? include
    : ["diary", "ideas", "web_research", "works", "bible"]
  ).filter((t) => ["diary", "ideas", "web_research", "works", "bible"].includes(t));

  const limit = Number.isFinite(limit_per_type)
    ? limit_per_type
    : DEFAULT_LIMIT_PER_TYPE;
  const previewLen = Number.isFinite(preview_chars)
    ? preview_chars
    : DEFAULT_PREVIEW_CHARS;

  const roots = {
    diary: DIARY_ROOT,
    ideas: IDEAS_ROOT,
    web_research: WEB_RESEARCH_ROOT,
    works: WORKS_ROOT,
    bible: BIBLE_ROOT,
  };

  try {
    const data = {};
    for (const t of types) {
      const root = roots[t];
      if (!root || !fs.existsSync(root)) {
        data[t] = [];
        continue;
      }

      const files = listFilesInRootWithMtime(root);
      const selected = [];
      for (const info of files) {
        if (selected.length >= limit) break;
        try {
          const { content } = safeReadFile(root, info.rel);
          const { meta, body } = parseFrontMatter(content);
          if (!matchesFilter(meta, body, info.rel, start_date, end_date, keyword)) {
            continue;
          }
          selected.push({
            type: t,
            rel_path: info.rel,
            path: info.abs,
            title: meta.title || path.basename(info.rel, path.extname(info.rel)),
            meta,
            excerpt: buildPreview(body, previewLen),
          });
        } catch (e) {
          // skip broken file
          continue;
        }
      }
      data[t] = selected;
    }

    return res.json({
      success: true,
      message: "ok",
      data,
      error: null,
    });
  } catch (e) {
    return res.status(500).json({
      success: false,
      message: "filter failed",
      data: null,
      error: e.message || String(e),
    });
  }
});

// ✅ 헬스체크용 라우트 추가
app.get("/health", (req, res) => {
  const exists = fs.existsSync(DIARY_ROOT);
  res.json({
    status: exists ? "ok" : "no_diary_root",
    diary_root: DIARY_ROOT,
    time: new Date().toISOString(),
    ideas_root: IDEAS_ROOT,
    web_research_root: WEB_RESEARCH_ROOT,
    works_root: WORKS_ROOT,
    bible_root: BIBLE_ROOT,
  });
});

app.listen(port, () => {
  console.log(
    `mcp-filesystem listening on ${port}, DIARY_ROOT=${DIARY_ROOT}, IDEAS_ROOT=${IDEAS_ROOT}, WEB_RESEARCH_ROOT=${WEB_RESEARCH_ROOT}, WORKS_ROOT=${WORKS_ROOT}, BIBLE_ROOT=${BIBLE_ROOT}`
  );
});
