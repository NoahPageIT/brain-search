// Build a local semantic index over the Obsidian second-brain vault.
// Uses transformers.js (all-MiniLM-L6-v2) - embeddings run on-device, no API.
// Run:  node index.js   (set NODE_OPTIONS=--use-system-ca if behind an SSL-inspecting proxy)
const fs = require('fs');
const path = require('path');

// Vault location: override with VAULT_PATH, else per-machine default
// (Windows keeps the vault in Projects; the Mac clone lives in Documents)
const os = require('os');
const VAULT = process.env.VAULT_PATH ||
  (process.platform === 'win32'
    ? 'C:\\Users\\mediabox\\Projects\\second-brain'
    : path.join(os.homedir(), 'Documents', 'SecondBrain'));
const OUT = path.join(__dirname, 'index.json');

function walk(dir, acc = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;            // skip .obsidian/.git/.trash
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walk(full, acc);
    else if (e.name.endsWith('.md')) acc.push(full);
  }
  return acc;
}

function chunkNote(text) {
  const paras = text.split(/\n\s*\n/).map(p => p.replace(/\s+/g, ' ').trim()).filter(p => p.length > 20);
  const chunks = [];
  let cur = '';
  for (const p of paras) {
    if ((cur + ' ' + p).length > 700) { if (cur) chunks.push(cur); cur = p; }
    else cur = cur ? cur + ' ' + p : p;
  }
  if (cur) chunks.push(cur);
  return chunks;
}

(async () => {
  const { pipeline } = await import('@xenova/transformers');
  console.log('Loading embedding model (first run downloads ~25MB, then cached)...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const files = walk(VAULT);
  const chunks = [];
  for (const f of files) {
    const raw = fs.readFileSync(f, 'utf8').replace(/^---[\s\S]*?---\s*/, ''); // strip frontmatter
    const title = path.basename(f, '.md');
    for (const c of chunkNote(raw)) chunks.push({ file: path.relative(VAULT, f), title, text: c });
  }
  console.log(`Found ${files.length} notes -> ${chunks.length} chunks. Embedding...`);
  for (let i = 0; i < chunks.length; i++) {
    const out = await extractor(chunks[i].text, { pooling: 'mean', normalize: true });
    chunks[i].vec = Array.from(out.data);
  }
  fs.writeFileSync(OUT, JSON.stringify({ model: 'all-MiniLM-L6-v2', dim: 384, builtAt: new Date().toISOString(), chunks }));
  console.log(`Indexed ${chunks.length} chunks -> ${OUT}`);
})().catch(e => { console.error('FAILED:', e.message); process.exit(1); });
