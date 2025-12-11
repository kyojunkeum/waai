import express from "express";
import fs from "fs";
import path from "path";
import url from "url";

const app = express();
const port = process.env.PORT || 7001;
const DIARY_ROOT = process.env.DIARY_ROOT || "/data/diary";

app.use(express.json());

function listFiles() {
  const results = [];
  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.isFile()) results.push(full);
    }
  }
  walk(DIARY_ROOT);
  return results;
}

app.get("/files", (req, res) => {
  const files = listFiles().map((p) => path.relative(DIARY_ROOT, p));
  res.json({ files });
});

app.get("/file", (req, res) => {
  const name = req.query.name;
  if (!name) return res.status(400).json({ error: "name query required" });
  const rootPath = path.resolve(DIARY_ROOT);
  const fullPath = path.resolve(rootPath, name);
  const relative = path.relative(rootPath, fullPath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    return res.status(400).json({ error: "invalid path" });
  }
  if (!fs.existsSync(fullPath))
    return res.status(404).json({ error: "file not found" });
  const content = fs.readFileSync(fullPath, "utf-8");
  res.json({ name, content });
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

// ✅ 헬스체크용 라우트 추가
app.get("/health", (req, res) => {
  const exists = fs.existsSync(DIARY_ROOT);
  res.json({
    status: exists ? "ok" : "no_diary_root",
    diary_root: DIARY_ROOT,
    time: new Date().toISOString(),
  });
});

app.listen(port, () => {
  console.log(`mcp-filesystem listening on ${port}, DIARY_ROOT=${DIARY_ROOT}`);
});
