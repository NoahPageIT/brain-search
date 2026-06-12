// Brain Search - local semantic search + RAG chat over the Obsidian second brain.
// Search uses on-device embeddings (no API). Chat uses a local Ollama model (also no API/cost).
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3005;
const PUBLIC = path.join(__dirname, 'public');
const INDEX = JSON.parse(fs.readFileSync(path.join(__dirname, 'index.json'), 'utf8'));
const OLLAMA = 'http://127.0.0.1:11434';
const MODEL = process.env.OLLAMA_MODEL || 'llama3.2';
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const json = (res, o) => { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(o)); };

let extractor = null, ready = false;
(async () => {
  const { pipeline } = await import('@xenova/transformers');
  extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  ready = true;
  console.log('  Embedding model ready.');
})();

const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
async function embed(q) { const out = await extractor(q, { pooling: 'mean', normalize: true }); return Array.from(out.data); }
async function retrieve(q, k) {
  const qv = await embed(q);
  return INDEX.chunks.map(c => ({ score: cosine(qv, c.vec), title: c.title, file: c.file, text: c.text }))
    .sort((a, b) => b.score - a.score).slice(0, k);
}

http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;

  if (p === '/api/status') return json(res, { ready, chunks: INDEX.chunks.length, model: MODEL });

  if (p === '/api/search') {
    const q = url.searchParams.get('q') || '';
    if (!ready || !q.trim()) return json(res, []);
    return retrieve(q, 10).then(r => json(res, r));
  }

  if (p === '/api/chat' && req.method === 'POST') {
    let body = ''; req.on('data', d => body += d);
    req.on('end', async () => {
      const { q } = JSON.parse(body || '{}');
      const ctx = await retrieve(q, 5);
      const sources = ctx.map(c => ({ title: c.title, file: c.file }));
      try {
        const prompt =
          `You are Noah's second-brain assistant. Answer the question using ONLY the notes below. ` +
          `Cite the note titles you used in brackets. If the notes don't contain the answer, say so plainly.\n\n` +
          `NOTES:\n${ctx.map((c, i) => `[${i + 1}] (${c.title}) ${c.text}`).join('\n\n')}\n\n` +
          `QUESTION: ${q}\n\nANSWER:`;
        const r = await fetch(`${OLLAMA}/api/generate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: MODEL, prompt, stream: false }),
        });
        if (!r.ok) throw new Error('ollama http ' + r.status);
        const data = await r.json();
        json(res, { answer: data.response, sources });
      } catch (e) {
        json(res, { answer: null, sources,
          error: `Ollama not detected. To enable AI answers: install Ollama, then run "ollama pull ${MODEL}". (Semantic search works without it.)` });
      }
    });
    return;
  }

  let file = p === '/' ? 'index.html' : p.replace(/^\/+/, '');
  const full = path.join(PUBLIC, file);
  if (full.startsWith(PUBLIC) && fs.existsSync(full)) {
    res.writeHead(200, { 'Content-Type': MIME[path.extname(full)] || 'text/plain' });
    return fs.createReadStream(full).pipe(res);
  }
  res.writeHead(404); res.end('Not found');
}).listen(PORT, () => console.log(`\n  🧠 Brain Search → http://localhost:${PORT}  (${INDEX.chunks.length} chunks indexed)\n`));
