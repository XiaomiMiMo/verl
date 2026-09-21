/* @@VFS_CORE_START@@ */

const VF_ORDER = { exact: 0, approx: 1, stale: 2, binary: 3, unknown: 4 };

function vfsNew() { return { files: new Map(), report: [], cwd: "" }; }

function vfsFile(vfs, p) {
  let f = vfs.files.get(p);
  if (!f) { f = { content: null, status: "unknown", url: null, ops: 0, misses: 0, log: [] }; vfs.files.set(p, f); }
  return f;
}

function vfsLog(f, s) { f.log.push(s); if (f.log.length > 40) f.log.shift(); }

function degrade(f, status) { if (VF_ORDER[status] > VF_ORDER[f.status]) f.status = status; }

/* ---- path handling ---- */

function canonPath(p, cwd) {
  if (typeof p !== "string") return null;
  p = p.trim().replace(/^["']+|["']+$/g, "");
  if (!p || p === "/dev/null" || p.startsWith("/dev/") || p.startsWith("/proc/")) return null;
  if (/[*?[\]{}$`]/.test(p)) return null;          // globs / expansions: not a concrete path
  if (!p.startsWith("/") && cwd) p = cwd.replace(/\/+$/, "") + "/" + p;
  const abs = p.startsWith("/");
  const parts = [];
  for (const seg of p.split("/")) {
    if (!seg || seg === ".") continue;
    if (seg === "..") { parts.pop(); continue; }
    parts.push(seg);
  }
  let out = (abs ? "/" : "") + parts.join("/");
  out = out.replace(/^\/(workspace|repo|app|project)\//, ""); // sandbox roots -> relative
  return out || null;
}

/* Resolve a mutation target against existing files (tolerates cwd mismatch). */
function resolvePath(vfs, p) {
  if (p === null) return null;
  if (vfs.files.has(p)) return p;
  const cands = [];
  for (const k of vfs.files.keys())
    if (k.endsWith("/" + p) || p.endsWith("/" + k)) cands.push(k);
  if (cands.length === 1) return cands[0];
  if (!cands.length) {
    const base = p.split("/").pop();
    for (const k of vfs.files.keys()) if (k.split("/").pop() === base) cands.push(k);
    if (cands.length === 1) return cands[0];
  }
  return p; // fall back to the literal path (may create a new entry)
}

function isTmpPath(p) { return /^\/?(tmp|var\/tmp|private\/tmp)\//.test(p) || p.startsWith("/tmp"); }

/* ---- mutation ops ---- */

function opWrite(vfs, path, content, fidelity, note) {
  const p = canonPath(path, vfs.cwd); if (p === null) return;
  const f = vfsFile(vfs, p);
  f.content = content; f.status = fidelity || "exact"; f.ops++;
  vfsLog(f, "write " + (note || "") + " (" + content.length + " ch)");
}

function opAppend(vfs, path, content, fidelity, note) {
  const p = resolvePath(vfs, canonPath(path, vfs.cwd)); if (p === null) return;
  const f = vfsFile(vfs, p);
  if (f.content === null) { f.content = content; degrade(f, "approx"); } // unknown base
  else f.content += content;
  if (fidelity) degrade(f, fidelity);
  f.ops++; vfsLog(f, "append " + (note || "") + " (+" + content.length + " ch)");
}

function opDelete(vfs, path) {
  const p = resolvePath(vfs, canonPath(path, vfs.cwd)); if (p === null) return;
  if (vfs.files.delete(p)) vfs.report.push("rm " + p);
}

function opDeleteGlob(vfs, pattern) { // e.g. rm -f /workspace/*.png
  const rx = globToRe(pattern); if (!rx) return;
  for (const k of [...vfs.files.keys()])
    if (rx.test(k) || rx.test("/" + k)) { vfs.files.delete(k); vfs.report.push("rm(glob) " + k); }
}

function globToRe(pat) {
  if (!/[*?]/.test(pat)) return null;
  let s = pat.replace(/^\/(workspace|repo|app|project)\//, "");
  s = s.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]*").replace(/\?/g, "[^/]");
  try { return new RegExp("^" + s + "$"); } catch { return null; }
}

function opRename(vfs, from, to) {
  const a = resolvePath(vfs, canonPath(from, vfs.cwd)), b = canonPath(to, vfs.cwd);
  if (a === null || b === null) return;
  const f = vfs.files.get(a);
  if (!f) { opStale(vfs, b, "mv from unknown " + a); return; }
  vfs.files.delete(a); vfs.files.set(b, f); vfsLog(f, "mv " + a + " -> " + b);
}

function opCopy(vfs, from, to) {
  const a = resolvePath(vfs, canonPath(from, vfs.cwd)), b = canonPath(to, vfs.cwd);
  if (a === null || b === null) return;
  const f = vfs.files.get(a);
  if (!f) { opStale(vfs, b, "cp from unknown " + a); return; }
  vfs.files.set(b, { ...f, log: [...f.log, "cp from " + a] });
}

function opBinary(vfs, path, url, note) {
  const p = canonPath(path, vfs.cwd); if (p === null) return;
  const f = vfsFile(vfs, p);
  f.content = null; f.status = "binary"; f.ops++;
  if (url && /^https?:\/\//i.test(url) && !/localhost|127\.0\.0\.1|0\.0\.0\.0/.test(url)) f.url = url;
  vfsLog(f, "binary " + (note || "") + (f.url ? " <- " + f.url : ""));
}

function opStale(vfs, path, note) {
  const p = resolvePath(vfs, canonPath(path, vfs.cwd)); if (p === null) return;
  const f = vfsFile(vfs, p);
  degrade(f, f.content === null ? "unknown" : "stale");
  vfsLog(f, "opaque: " + (note || "?"));
  vfs.report.push("opaque mutation on " + p + ": " + (note || "?"));
}

/* ---- literal edit with fuzzy fallback ---- */

function opEditLiteral(vfs, path, oldStr, newStr, replaceAll, note) {
  const p = resolvePath(vfs, canonPath(path, vfs.cwd)); if (p === null) return;
  const f = vfsFile(vfs, p);
  f.ops++;
  if (f.content === null) { degrade(f, "unknown"); f.misses++; vfsLog(f, "edit on unknown base"); return; }
  const r = fuzzyReplace(f.content, oldStr, newStr, replaceAll);
  if (r.ok) {
    f.content = r.content;
    if (r.fuzzy) degrade(f, "approx");
    vfsLog(f, "edit " + (note || "") + (r.fuzzy ? " (fuzzy)" : ""));
  } else {
    f.misses++; degrade(f, "approx");
    vfsLog(f, "edit MISS " + (note || "") + ": " + String(oldStr).slice(0, 60).replace(/\n/g, "\\n"));
    vfs.report.push("edit miss on " + p);
  }
}

function fuzzyReplace(content, oldStr, newStr, replaceAll) {
  if (typeof oldStr !== "string" || typeof newStr !== "string" || !oldStr.length)
    return { ok: false };
  if (content.includes(oldStr)) {
    return { ok: true, fuzzy: false, content: replaceAll
      ? content.split(oldStr).join(newStr)
      : content.replace(oldStr, () => newStr) };
  }
  const crlf = oldStr.replace(/\r\n/g, "\n");
  if (crlf !== oldStr && content.includes(crlf))
    return { ok: true, fuzzy: true, content: content.replace(crlf, () => newStr) };
  // line-trimmed sliding window: tolerates indentation drift
  const cl = content.split("\n"), ol = oldStr.split("\n");
  if (ol.length >= 1 && cl.length >= ol.length) {
    const oT = ol.map(s => s.trim());
    for (let i = 0; i + ol.length <= cl.length; i++) {
      let hit = true;
      for (let j = 0; j < ol.length; j++) if (cl[i + j].trim() !== oT[j]) { hit = false; break; }
      if (hit) {
        const before = cl.slice(0, i), after = cl.slice(i + ol.length);
        return { ok: true, fuzzy: true, content: [...before, ...newStr.split("\n"), ...after].join("\n") };
      }
    }
  }
  return { ok: false };
}

/* ---- context-hunk applier (shared by V4A patches and unified diffs) ---- */
/* hunk = [{tag:' '|'-'|'+', text}] — find the (ctx+del) line sequence, swap in (ctx+add) */

function applyHunks(content, hunks) {
  let lines = content.split("\n");
  let searchFrom = 0, misses = 0, fuzzy = false;
  for (const hunk of hunks) {
    const find = hunk.filter(h => h.tag !== "+").map(h => h.text);
    const repl = hunk.filter(h => h.tag !== "-").map(h => h.text);
    if (!find.length) { lines = lines.concat(repl); fuzzy = true; continue; }
    let idx = matchLines(lines, find, searchFrom, false);
    if (idx < 0) idx = matchLines(lines, find, 0, false);
    if (idx < 0) {
      idx = matchLines(lines, find, searchFrom, true);
      if (idx < 0) idx = matchLines(lines, find, 0, true);
      if (idx >= 0) fuzzy = true;
    }
    if (idx < 0) { misses++; continue; }
    lines.splice(idx, find.length, ...repl);
    searchFrom = idx + repl.length;
  }
  // real patch tools (codex apply_patch, git apply) are atomic: any unmatched
  // hunk rejects the whole patch, so mirroring that keeps our state honest
  if (misses) return { content, misses, fuzzy, rejected: true };
  return { content: lines.join("\n"), misses, fuzzy, rejected: false };
}

function matchLines(hay, needle, from, trimmed) {
  const t = trimmed ? (s => s.trim()) : (s => s);
  const n = needle.map(t);
  outer: for (let i = from; i + needle.length <= hay.length; i++) {
    for (let j = 0; j < needle.length; j++) if (t(hay[i + j]) !== n[j]) continue outer;
    return i;
  }
  return -1;
}

/* ---- shell command handling ---- */

/* Pull a runnable command string out of tool args of many shapes:
   {cmd}, {command}, {script}, ["bash","-lc","…"], "/bin/bash -lc '…'" wrappers. */
function extractCommand(args) {
  let c = null;
  if (typeof args === "string") c = args;
  else if (args && typeof args === "object")
    c = args.cmd ?? args.command ?? args.script ?? args.bash_command ?? args.code ?? null;
  if (Array.isArray(c)) {
    const i = c.findIndex(x => /^-l?c l?$|^-l?c$|^-cl?$/.test(String(x)));
    c = i >= 0 && i + 1 < c.length ? String(c[i + 1]) : c.map(String).join(" ");
  }
  if (typeof c !== "string") return null;
  return unwrapShell(c);
}

function unwrapShell(cmd) {
  for (let depth = 0; depth < 3; depth++) {
    const m = cmd.match(/^\s*(?:\/(?:usr\/)?bin\/)?(?:env\s+)?(?:ba|z|da)?sh\s+(?:-[a-z]+\s+)*(["'])([\s\S]*)\1\s*$/);
    if (!m) break;
    let inner = m[2];
    if (m[1] === '"') inner = inner.replace(/\\([$`"\\\n])/g, "$1");
    cmd = inner;
  }
  return cmd;
}

/* Split a shell command into heredoc documents and residual shell lines.
   This isolation is the #1 robustness lever: heredoc bodies (file contents)
   must never be scanned for shell idioms. */
function splitHeredocs(cmd) {
  const lines = cmd.split("\n");
  const docs = [], shellLines = [];
  const INTRO = /<<-?\s*(?:'([A-Za-z_][A-Za-z0-9_]*)'|"([A-Za-z_][A-Za-z0-9_]*)"|\\?([A-Za-z_][A-Za-z0-9_]*))/g;
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    INTRO.lastIndex = 0;
    const intros = [];
    let m;
    while ((m = INTRO.exec(line)) !== null) {
      const delim = m[1] || m[2] || m[3];
      const quoted = !!(m[1] || m[2]);
      // confirm a terminator exists ahead; otherwise it's not a real heredoc (e.g. `a<<b` math)
      let term = -1;
      for (let j = i + 1 + intros.reduce((s, x) => s + x.bodyLen + 1, 0); j < lines.length; j++) {
        if (lines[j].trim() === delim) { term = j; break; }
      }
      if (term < 0) continue;
      const startBody = i + 1 + intros.reduce((s, x) => s + x.bodyLen + 1, 0);
      intros.push({ delim, quoted, bodyLen: term - startBody, startBody });
    }
    if (!intros.length) { shellLines.push(line); i++; continue; }

    const residual = line.replace(INTRO, " ");
    for (const it of intros) {
      const body = lines.slice(it.startBody, it.startBody + it.bodyLen).join("\n");
      docs.push({ intro: line, residual, body: body + "\n", quoted: it.quoted });
    }
    shellLines.push(residual);
    const last = intros[intros.length - 1];
    i = last.startBody + last.bodyLen + 1; // skip past final terminator
  }
  return { docs, shellLines };
}

/* Classify a heredoc intro line: what consumes the body? */
function classifyHeredoc(introLine) {
  const seg = introLine.split(/(?<!\|)\|(?!\|)|&&|;/).find(s => /<</.test(s)) || introLine;
  const target = redirectTarget(seg) || redirectTarget(introLine);
  if (/\bapply_patch\b/.test(seg)) return { kind: "apply_patch" };
  if (/\bgit\s+apply\b|\bpatch\b\s+(-p\d|\S*--)/.test(seg)) return { kind: "unidiff" };
  if (/\bpython[0-9.]*\b/.test(seg)) return { kind: "python", target };
  if (/\bnode(js)?\b/.test(seg)) return { kind: "node", target };
  if (/\b(ba|z|da)?sh\b\s*(<<|-s)/.test(seg)) return { kind: "shell", target };
  const tee = seg.match(/\btee\b\s+(-a\s+)?((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)/);
  if (tee) return { kind: "file", target: tee[2], append: !!tee[1] };
  if (target) return { kind: "file", target, append: /(^|[^>])>>\s*\S/.test(seg) };
  return { kind: "ignore" };
}

function redirectTarget(s) {
  // last `> target` / `>> target`, ignoring fd dups (2>&1), /dev/null, and `<<`
  let best = null;
  const re = /(?:^|[^<>])>{1,2}\s*((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)/g;
  let m;
  while ((m = re.exec(s)) !== null) {
    const t = m[1];
    if (/^&/.test(t) || t === "/dev/null" || /^\/dev\//.test(t)) continue;
    best = t;
  }
  return best;
}

/* Scan one residual shell line for file-affecting idioms. */
function scanShellLine(vfs, line, execScript) {
  if (!line || /^\s*#/.test(line)) return;

  // split on && ; | boundaries conservatively (quotes rare on these lines)
  for (let seg of line.split(/&&|;|\|\|/)) {
    seg = seg.trim();
    if (!seg) continue;

    let m;
    if ((m = seg.match(/^cd\s+((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)$/))) {
      const d = m[1].replace(/^["']|["']$/g, "");
      if (!/[$`]/.test(d)) vfs.cwd = d.startsWith("/") ? d : (vfs.cwd ? vfs.cwd + "/" + d : d);
      continue;
    }
    if ((m = seg.match(/^mv\s+(?:-[a-zA-Z]+\s+)*(\S+)\s+(\S+)$/))) { opRename(vfs, m[1], m[2]); continue; }
    if ((m = seg.match(/^cp\s+(?:-[a-zA-Z]+\s+)*(\S+)\s+(\S+)$/))) { opCopy(vfs, m[1], m[2]); continue; }
    if ((m = seg.match(/^rm\s+((?:-[a-zA-Z]+\s+)*)(.+)$/))) {
      for (const t of m[2].split(/\s+/)) {
        if (t.startsWith("-")) continue;
        if (/[*?]/.test(t)) opDeleteGlob(vfs, t); else opDelete(vfs, t);
      }
      continue;
    }
    if ((m = seg.match(/^touch\s+(.+)$/))) {
      for (const t of m[1].split(/\s+/)) {
        if (t.startsWith("-")) continue;
        const p = canonPath(t, vfs.cwd);
        if (p && !vfs.files.has(p)) opWrite(vfs, t, "", "exact", "touch");
      }
      continue;
    }
    // downloads: curl -o T URL | curl URL > T | wget -O T URL
    if (/\bcurl\b/.test(seg)) {
      const out = seg.match(/\s(?:-o|--output)\s+((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)/);
      const url = seg.match(/(["']?)(https?:\/\/[^\s"'|;&<>)]+)\1/);
      const redir = redirectTarget(seg);
      const tgt = out ? out[1] : redir;
      if (tgt && !/\.(log|txt|json|pid)$/i.test(tgt))
        opBinary(vfs, tgt, url ? url[2] : null, "curl");
      continue;
    }
    if (/\bwget\b/.test(seg)) {
      const out = seg.match(/\s-O\s+((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)/);
      const url = seg.match(/(["']?)(https?:\/\/[^\s"'|;&<>)]+)\1/);
      if (out) opBinary(vfs, out[1], url ? url[2] : null, "wget");
      continue;
    }
    // image producers: convert/magick/rsvg-convert/inkscape … out.img
    if ((m = seg.match(/^(?:convert|magick|mogrify|rsvg-convert|inkscape|ffmpeg)\b.*?((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+\.(?:png|jpe?g|gif|webp|ico|avif))\s*$/i))) {
      opBinary(vfs, m[1], null, "imagetool"); continue;
    }
    // echo/printf literal redirects
    if ((m = seg.match(/^(echo|printf)\s+(.*?)\s*(>>?)\s*((?:"[^"]*"|'[^']*'|[^\s<>|;&()])+)\s*$/))) {
      const [, cmd0, argstr, op, tgt] = m;
      const lit = shellLiteral(argstr, cmd0 === "echo");
      if (lit === null) { opStale(vfs, tgt, cmd0 + " with expansions"); continue; }
      if (op === ">") opWrite(vfs, tgt, lit, "exact", cmd0);
      else opAppend(vfs, tgt, lit, "exact", cmd0);
      continue;
    }
    // sed -i: literal s/…/…/ only; anything fancier marks the file stale
    if (/\bsed\b.*\s-i/.test(seg)) {
      const expr = seg.match(/(?:-e\s+)?(["'])(s.[\s\S]+?)\1/);
      const files = seg.match(/\s((?:[^\s<>|;&()'"-][^\s<>|;&()'"]*\.(?:html?|css|js|mjs|cjs|json|svg|md|txt|xml|py)))/g);
      const targets = (files || []).map(s => s.trim());
      if (expr && targets.length) {
        const sm = expr[2].match(/^s(.)((?:\\.|(?!\1).)*)\1((?:\\.|(?!\1).)*)\1(g?)\s*$/);
        const litOk = sm && !/[\\.^$*+?()[\]{}|]/.test(sm[2].replace(/\\[/.]/g, ""));
        for (const t of targets) {
          if (litOk) opEditLiteral(vfs, t, sm[2].replace(/\\(.)/g, "$1"), sm[3].replace(/\\(.)/g, "$1"), sm[4] === "g", "sed");
          else opStale(vfs, t, "sed -i (regex)");
        }
      }
      continue;
    }
    // inline interpreters: python -c '…' / node -e '…'
    if ((m = seg.match(/\bpython[0-9.]*\s+(?:-[A-Za-z]+\s+)*-c\s+(["'])([\s\S]*)\1/))) { execScript("python", m[2]); continue; }
    if ((m = seg.match(/\bnode\s+(?:-[A-Za-z]+\s+)*(?:-e|--eval)\s+(["'])([\s\S]*)\1/))) { execScript("node", m[2]); continue; }
  }
}

/* echo/printf arg list -> literal string, or null if it needs shell expansion */
function shellLiteral(argstr, addNewline) {
  let s = argstr.trim(), out = "";
  let esc = /^-e\s+/.test(s);
  s = s.replace(/^-[en]\s+/, "").replace(/^-[en]\s+/, "");
  while (s.length) {
    let m;
    if ((m = s.match(/^'([^']*)'/))) { out += m[1]; s = s.slice(m[0].length); }
    else if ((m = s.match(/^"((?:\\.|[^"\\])*)"/))) {
      const inner = m[1];
      if (/[$`]/.test(inner.replace(/\\[$`]/g, ""))) return null;
      out += inner.replace(/\\([$`"\\])/g, "$1"); s = s.slice(m[0].length);
    }
    else if ((m = s.match(/^[^\s"'$`\\]+/))) { out += m[0]; s = s.slice(m[0].length); }
    else if (/^\s+/.test(s)) { out += " "; s = s.replace(/^\s+/, ""); }
    else return null; // $var, backtick, lone backslash…
  }
  if (out.includes("%")) {
    if (/%[^%]/.test(out)) return null; // printf directives are not emulated
    out = out.replace(/%%/g, "%");
  }
  if (esc || !addNewline) // printf always processes escapes; echo only with -e
    out = out.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\r/g, "\r").replace(/\\\\/g, "\\");
  return addNewline ? out + "\n" : out;
}

/* ---- python string literals ---- */
/* Parses (a concatenation of) python string literals. Returns null when the
   expression is not a compile-time constant (f-string with {}, variables…). */

function pyLiteral(expr) {
  let s = expr.trim(), out = "", any = false;
  // strip a single fully-wrapping paren group
  while (s.startsWith("(") && s.endsWith(")") && balancedParens(s.slice(1, -1))) s = s.slice(1, -1).trim();
  while (s.length) {
    const m = s.match(/^([rRbBuUfF]{0,3})('''|"""|'|")/);
    if (!m) return null;
    const prefix = m[1].toLowerCase(), q = m[2];
    const raw = prefix.includes("r"), fstr = prefix.includes("f");
    let i = m[0].length, val = "";
    for (;;) {
      if (i >= s.length) return null; // unterminated
      if (s.startsWith(q, i)) { i += q.length; break; }
      const ch = s[i];
      if (ch === "\\" && !raw) {
        const e = s[i + 1];
        const map = { n: "\n", t: "\t", r: "\r", "0": "\0", "\\": "\\", "'": "'", '"': '"', a: "\x07", b: "\b", f: "\f", v: "\v", "\n": "" };
        if (e === "x" && /^[0-9a-fA-F]{2}/.test(s.slice(i + 2))) { val += String.fromCharCode(parseInt(s.slice(i + 2, i + 4), 16)); i += 4; continue; }
        if (e === "u" && /^[0-9a-fA-F]{4}/.test(s.slice(i + 2))) { val += String.fromCharCode(parseInt(s.slice(i + 2, i + 6), 16)); i += 6; continue; }
        if (e in map) { val += map[e]; i += 2; continue; }
        val += ch + e; i += 2; continue;
      }
      if (ch === "\\" && raw) { val += ch + (s[i + 1] ?? ""); i += 2; continue; }
      if (q.length === 1 && ch === "\n") return null; // newline in short string
      val += ch; i++;
    }
    if (fstr) {
      if (/(?<!\{)\{(?!\{)/.test(val)) return null; // real interpolation
      val = val.replace(/\{\{/g, "{").replace(/\}\}/g, "}");
    }
    out += val; any = true;
    s = s.slice(i).trim();
    if (s.startsWith("+")) s = s.slice(1).trim(); // explicit concat; implicit concat = just continue
    else if (s.startsWith(",")) return null;      // arg list, not a single string
  }
  return any ? out : null;
}

function balancedParens(s) {
  let d = 0, q = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === "\\") i++; else if (c === q) q = null; continue; }
    if (c === "'" || c === '"') q = c;
    else if (c === "(") d++;
    else if (c === ")") { d--; if (d < 0) return false; }
  }
  return d === 0;
}

/* Slice the argument text of `name(` starting at the char after '(' — string-aware. */
function sliceCallArg(body, openIdx) {
  let d = 1, q = null, triple = null;
  for (let i = openIdx + 1; i < body.length; i++) {
    const c = body[i];
    if (triple) { if (body.startsWith(triple, i)) { i += 2; triple = null; } continue; }
    if (q) { if (c === "\\") i++; else if (c === q) q = null; continue; }
    if (c === "'" || c === '"') {
      const t = body.slice(i, i + 3);
      if (t === "'''" || t === '"""') { triple = t; i += 2; } else q = c;
      continue;
    }
    if (c === "(" || c === "[" || c === "{") d++;
    else if (c === ")" || c === "]" || c === "}") { d--; if (d === 0) return { arg: body.slice(openIdx + 1, i), end: i }; }
  }
  return null;
}

/* ---- python script bodies (from `python3 - <<PY` heredocs / -c) ---- */

function parsePythonScript(vfs, body) {
  const pathVars = new Map();  // var -> path       (p = Path('x'))
  const strVars = new Map();   // var -> {path, edits:[[old,new,count]]} | null (opaque)
  const urls = [];
  let m;
  const urlRe = /(?:urlopen|urlretrieve|requests\.get)\(\s*((?:[rbuf]{0,3}["'][^"']*["']\s*[+,]?\s*)+)/g;
  while ((m = urlRe.exec(body)) !== null) {
    const lit = pyLiteral(m[1].replace(/,[\s\S]*$/, ""));
    if (lit && /^https?:\/\//.test(lit)) urls.push(lit);
  }
  const urRe = /urlretrieve\(([^)]*)\)/g; // urlretrieve(url, target)
  while ((m = urRe.exec(body)) !== null) {
    const parts = topLevelSplit(m[1]);
    if (parts.length >= 2) {
      const u = pyLiteral(parts[0]), t = pyLiteral(parts[1]);
      if (t) opBinary(vfs, t, u, "urlretrieve");
    }
  }

  /* sequential pass: var state must be tracked in program order —
     `p=Path('a') … p.write_text(s) … p=Path('b') … p.write_text(s)` */
  const lines = body.split("\n");
  const offs = []; { let o = 0; for (const l of lines) { offs.push(o); o += l.length + 1; } }
  const STR_ARG = "((?:[rbuf]{0,3}(?:'[^']*'|\"[^\"]*\")\\s*\\+?\\s*)+)";
  const wRe = new RegExp("(?:\\b(\\w+)|(?:pathlib\\.)?Path\\(\\s*" + STR_ARG + "\\s*\\))\\.write_(text|bytes)\\s*\\(", "g");
  const oRe = new RegExp("open\\(\\s*" + STR_ARG + "\\s*,\\s*['\"]([wa])b?['\"]\\s*\\)", "g");

  for (let li = 0; li < lines.length; li++) {
    const raw = lines[li], line = raw.trim();
    let mm, consumedTo = -1;

    if ((mm = line.match(/^(\w+)\s*=\s*(?:pathlib\.)?Path\(\s*([^)]*)\)\s*$/))) {
      const p = pyLiteral(mm[2]);
      if (p) pathVars.set(mm[1], p); else pathVars.delete(mm[1]);
      continue;
    }
    if ((mm = line.match(/^(\w+)\s*=\s*(\w+)\.read_text\(/))) {
      const p = pathVars.get(mm[2]);
      strVars.set(mm[1], p ? { path: p, edits: [] } : null);
      continue;
    }
    if ((mm = line.match(/^(\w+)\s*=\s*(?:pathlib\.)?Path\(\s*([^)]*?)\)\.read_text\(/))) {
      const p = pyLiteral(mm[2]);
      strVars.set(mm[1], p ? { path: p, edits: [] } : null);
      continue;
    }
    if ((mm = line.match(/^(\w+)\s*=\s*(\w+)\.replace\(/))) {
      const openIdx = offs[li] + raw.indexOf(".replace(") + ".replace".length;
      const call = sliceCallArg(body, openIdx);
      if (call) while (li + 1 < lines.length && offs[li + 1] <= call.end) li++;
      if (mm[1] !== mm[2]) { if (strVars.has(mm[1])) strVars.set(mm[1], null); continue; }
      const sv = strVars.get(mm[2]);
      if (!sv) continue;
      if (!call || !sv.edits) { sv.edits = null; continue; }
      const parts = topLevelSplit(call.arg);
      const a = parts.length >= 2 ? pyLiteral(parts[0]) : null;
      const b = parts.length >= 2 ? pyLiteral(parts[1]) : null;
      if (a !== null && b !== null) sv.edits.push([a, b, parts[2] ? parseInt(parts[2], 10) || 1 : Infinity]);
      else sv.edits = null; // non-literal replace -> opaque
      continue;
    }

    // writes on this line (arguments may span lines — slice from the full body)
    wRe.lastIndex = 0;
    while ((mm = wRe.exec(raw)) !== null) {
      const path = mm[1] ? pathVars.get(mm[1]) : pyLiteral(mm[2]);
      const call = sliceCallArg(body, offs[li] + mm.index + mm[0].length - 1);
      if (call) consumedTo = Math.max(consumedTo, call.end);
      if (!path) continue;
      if (mm[3] === "bytes") { opBinary(vfs, path, urls.length === 1 ? urls[0] : null, "write_bytes"); continue; }
      if (!call) { opStale(vfs, path, "write_text"); continue; }
      pyApplyWrite(vfs, path, call.arg, strVars, false);
    }
    oRe.lastIndex = 0;
    while ((mm = oRe.exec(raw)) !== null) {
      const path = pyLiteral(mm[1]); if (!path) continue;
      const from = offs[li] + mm.index;
      const rest = body.slice(from, from + 2500);
      const w = rest.match(/\.write\s*\(/);
      if (!w) { opStale(vfs, path, "open(" + mm[2] + ") no visible write"); continue; }
      const call = sliceCallArg(body, from + w.index + w[0].length - 1);
      if (call) consumedTo = Math.max(consumedTo, call.end);
      if (!call) { opStale(vfs, path, "open write"); continue; }
      pyApplyWrite(vfs, path, call.arg, strVars, mm[2] === "a");
    }
    // any other reassignment of a tracked string var -> opaque
    if (consumedTo < 0 && (mm = line.match(/^(\w+)\s*=[^=]/)) && strVars.has(mm[1])) {
      const sv = strVars.get(mm[1]); if (sv) sv.edits = null;
    }
    if (consumedTo >= 0) while (li + 1 < lines.length && offs[li + 1] <= consumedTo) li++;
  }

  // order-insensitive extras (inline literal paths only)
  const jRe = /json\.dump\([^,]+,\s*open\(\s*((?:[rbuf]{0,3}(?:'[^']*'|"[^"]*")\s*)+)/g;
  while ((m = jRe.exec(body)) !== null) { const p = pyLiteral(m[1]); if (p) opStale(vfs, p, "json.dump"); }
  const sRe = /\.save\(\s*((?:[rbuf]{0,3}(?:'[^']*'|"[^"]*")\s*)+)[,)]/g;
  while ((m = sRe.exec(body)) !== null) { const p = pyLiteral(m[1]); if (p && /\.(png|jpe?g|gif|webp|ico|svg|avif)$/i.test(p)) opBinary(vfs, p, null, "PIL save"); }
  const dRe = /(?:os\.remove|os\.unlink)\(\s*((?:[rbuf]{0,3}(?:'[^']*'|"[^"]*")\s*)+)\)|Path\(\s*((?:[rbuf]{0,3}(?:'[^']*'|"[^"]*")\s*)+)\)\.unlink\(/g;
  while ((m = dRe.exec(body)) !== null) { const p = pyLiteral(m[1] || m[2]); if (p) opDelete(vfs, p); }
  const mvRe = /(?:shutil\.move|os\.rename|os\.replace)\(([^)]*)\)/g;
  while ((m = mvRe.exec(body)) !== null) {
    const parts = topLevelSplit(m[1]);
    const a = parts[0] && pyLiteral(parts[0]), b = parts[1] && pyLiteral(parts[1]);
    if (a && b) opRename(vfs, a, b);
  }
  const cpRe = /shutil\.copy2?\(([^)]*)\)/g;
  while ((m = cpRe.exec(body)) !== null) {
    const parts = topLevelSplit(m[1]);
    const a = parts[0] && pyLiteral(parts[0]), b = parts[1] && pyLiteral(parts[1]);
    if (a && b) opCopy(vfs, a, b);
  }
}

/* write `argExpr` to `path`: literal -> full write; tracked read+replace var -> edits */
function pyApplyWrite(vfs, path, argExpr, strVars, append) {
  const argParts = topLevelSplit(argExpr);
  const lit = argParts.length ? pyLiteral(argParts[0]) : null;
  if (lit !== null) {
    if (append) opAppend(vfs, path, lit, "exact", "py write");
    else opWrite(vfs, path, lit, "exact", "py write");
    return;
  }
  const varName = (argParts[0] || "").trim();
  const sv = strVars.get(varName);
  if (sv && sv.path) {
    if (sv.edits === null) { opStale(vfs, path, "py opaque transform of " + varName); return; }
    if (sv.path !== path) {
      const src = vfs.files.get(resolvePath(vfs, canonPath(sv.path, vfs.cwd)));
      if (src && src.content !== null) opWrite(vfs, path, src.content, "approx", "py copy via " + varName);
      else { opStale(vfs, path, "py write from other file"); return; }
    }
    for (const [a, b, count] of sv.edits) {
      if (count === Infinity) opEditLiteral(vfs, path, a, b, true, "py replace");
      else for (let k = 0; k < count; k++) opEditLiteral(vfs, path, a, b, false, "py replace#" + count);
    }
    sv.edits = []; // applied; a later write_text(s) of the same var must not re-apply
  } else opStale(vfs, path, "py write(non-literal)");
}

/* split "a, b, c" at top level (string/paren aware) */
function topLevelSplit(s) {
  const parts = []; let d = 0, q = null, triple = null, cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (triple) { cur += c; if (s.startsWith(triple, i)) { cur += s.slice(i + 1, i + 3); i += 2; triple = null; } continue; }
    if (q) { cur += c; if (c === "\\") { cur += s[i + 1] ?? ""; i++; } else if (c === q) q = null; continue; }
    if (c === "'" || c === '"') {
      const t = s.slice(i, i + 3);
      if (t === "'''" || t === '"""') { triple = t; cur += t; i += 2; } else { q = c; cur += c; }
      continue;
    }
    if ("([{".includes(c)) d++;
    if (")]}".includes(c)) d--;
    if (c === "," && d === 0) { parts.push(cur); cur = ""; continue; }
    cur += c;
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}

/* ---- node script bodies ---- */

function parseNodeScript(vfs, body) {
  let m;
  const wRe = /(?:fs\.|fsp\.|fs\.promises\.)?writeFile(?:Sync)?\s*\(/g;
  while ((m = wRe.exec(body)) !== null) {
    const call = sliceCallArg(body, m.index + m[0].length - 1);
    if (!call) continue;
    const parts = topLevelSplit(call.arg);
    if (parts.length < 2) continue;
    const path = jsLiteral(parts[0]);
    if (!path) continue;
    const content = jsLiteral(parts[1]);
    if (content !== null) opWrite(vfs, path, content, "exact", "fs.writeFile");
    else opStale(vfs, path, "fs.writeFile(non-literal)");
  }
  const dRe = /fs\.(?:unlinkSync|rmSync|unlink|rm)\s*\(\s*(['"`][^'"`]*['"`])/g;
  while ((m = dRe.exec(body)) !== null) { const p = jsLiteral(m[1]); if (p) opDelete(vfs, p); }
  const sRe = /screenshot\s*\(\s*\{[^}]*path\s*:\s*(['"`][^'"`]*['"`])/g;
  while ((m = sRe.exec(body)) !== null) { const p = jsLiteral(m[1]); if (p) opBinary(vfs, p, null, "screenshot"); }
}

function jsLiteral(expr) {
  let s = expr.trim(), out = "", any = false;
  while (s.length) {
    // String.raw`…`: template with raw semantics — backslashes stay verbatim
    let raw = false;
    const rm = s.match(/^String\.raw\s*(?=`)/);
    if (rm) { raw = true; s = s.slice(rm[0].length); }
    const q = s[0];
    if (q !== "'" && q !== '"' && q !== "`") return any && !s.trim() ? out : null;
    if (raw && q !== "`") return null;
    let i = 1, val = "";
    for (;;) {
      if (i >= s.length) return null;
      const c = s[i];
      if (c === "\\") {
        if (raw) { val += c + (s[i + 1] ?? ""); i += 2; continue; }
        const e = s[i + 1];
        const map = { n: "\n", t: "\t", r: "\r", "\\": "\\", "'": "'", '"': '"', "`": "`", "$": "$", "0": "\0", "\n": "" };
        if (e === "u" && s[i + 2] === "{") { const close = s.indexOf("}", i); val += String.fromCodePoint(parseInt(s.slice(i + 3, close), 16)); i = close + 1; continue; }
        if (e === "u") { val += String.fromCharCode(parseInt(s.slice(i + 2, i + 6), 16)); i += 6; continue; }
        if (e === "x") { val += String.fromCharCode(parseInt(s.slice(i + 2, i + 4), 16)); i += 4; continue; }
        val += (e in map ? map[e] : e); i += 2; continue;
      }
      if (c === q) { i++; break; }
      if (q === "`" && c === "$" && s[i + 1] === "{") return null; // interpolation
      if (q !== "`" && c === "\n") return null;
      val += c; i++;
    }
    out += val; any = true;
    s = s.slice(i).trim();
    if (s.startsWith("+")) s = s.slice(1).trim();
    else break;
  }
  return any && !s.length ? out : (any ? out : null);
}

/* ---- programmatic tool calling ----
   Some scaffolds emit a JS orchestration program as the tool argument:
     const patch = "*** Begin Patch\n…";
     text(await tools.apply_patch(patch));
     const r = await tools.exec_command({cmd:"…", workdir:"/workspace"});
   Statically resolve string-constant bindings and re-dispatch every
   `tools.NAME(…)` call site through applyToolCall. */

/* Mask of "executable code" positions: 0 inside string literals & comments,
   so binding/call scans never fire on code-looking text within patch strings.
   (Regex literals containing quotes can locally confuse the mask — rare and
   non-fatal: affected calls land in the report as unresolved.) */
function jsCodeMask(src) {
  const mask = new Uint8Array(src.length).fill(1);
  let i = 0, q = null;
  while (i < src.length) {
    const c = src[i];
    if (q) {
      mask[i] = 0;
      if (c === "\\") { if (i + 1 < src.length) mask[i + 1] = 0; i += 2; continue; }
      if (c === q) q = null;
      i++; continue;
    }
    if (c === "'" || c === '"' || c === "`") { q = c; mask[i] = 0; i++; continue; }
    if (c === "/" && src[i + 1] === "/") { while (i < src.length && src[i] !== "\n") mask[i++] = 0; continue; }
    if (c === "/" && src[i + 1] === "*") {
      mask[i] = 0; mask[i + 1] = 0; i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) mask[i++] = 0;
      if (i < src.length) { mask[i] = 0; mask[i + 1] = 0; i += 2; }
      continue;
    }
    i++;
  }
  return mask;
}

function jsSliceCall(body, openIdx) { // like sliceCallArg, but backtick-aware
  let d = 1, q = null;
  for (let i = openIdx + 1; i < body.length; i++) {
    const c = body[i];
    if (q) { if (c === "\\") i++; else if (c === q) q = null; continue; }
    if (c === "'" || c === '"' || c === "`") { q = c; continue; }
    if (c === "(" || c === "[" || c === "{") d++;
    else if (c === ")" || c === "]" || c === "}") { d--; if (d === 0) return { arg: body.slice(openIdx + 1, i), end: i }; }
  }
  return null;
}

function jsStatementEnd(src, from) {
  let d = 0, q = null;
  for (let i = from; i < src.length; i++) {
    const c = src[i];
    if (q) { if (c === "\\") i++; else if (c === q) q = null; continue; }
    if (c === "'" || c === '"' || c === "`") { q = c; continue; }
    if ("([{".includes(c)) d++;
    else if (")]}".includes(c)) d--;
    else if ((c === ";" || c === "\n") && d <= 0) return i;
  }
  return src.length;
}

function jsSplitPlus(s) { // split on top-level "+", string-aware
  const parts = []; let d = 0, q = null, cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { cur += c; if (c === "\\") { cur += s[i + 1] ?? ""; i++; } else if (c === q) q = null; continue; }
    if (c === "'" || c === '"' || c === "`") { q = c; cur += c; continue; }
    if ("([{".includes(c)) d++;
    if (")]}".includes(c)) d--;
    if (c === "+" && d === 0 && s[i + 1] !== "+" && s[i - 1] !== "+") { parts.push(cur); cur = ""; continue; }
    cur += c;
  }
  parts.push(cur);
  return parts;
}

function jsBalancedWrap(s) { // "(…)" wrapping the whole expression?
  if (!s.startsWith("(") || !s.endsWith(")")) return false;
  let d = 0, q = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === "\\") i++; else if (c === q) q = null; continue; }
    if (c === "'" || c === '"' || c === "`") { q = c; continue; }
    if (c === "(") d++;
    else if (c === ")") { d--; if (d === 0) return i === s.length - 1; }
  }
  return false;
}

function jsResolveExpr(expr, bindings) { // string value of literal/ident/concat, or null
  let out = "";
  for (let part of jsSplitPlus(expr)) {
    part = part.trim();
    while (jsBalancedWrap(part)) part = part.slice(1, -1).trim();
    const lit = jsLiteral(part);
    if (lit !== null) { out += lit; continue; }
    if (/^[\w$]+$/.test(part) && bindings.has(part)) { out += bindings.get(part); continue; }
    return null;
  }
  return out;
}

function parseJsObjectArg(text, bindings) {
  const t = text.trim();
  if (!t.startsWith("{") || !t.includes("}")) return null;
  const out = {};
  for (const part of topLevelSplit(t.slice(1, t.lastIndexOf("}")))) {
    const mm = part.match(/^\s*(?:(['"])([\s\S]*?)\1|([\w$]+))\s*:\s*([\s\S]+)$/);
    if (!mm) continue; // spread / shorthand / method — not needed for cmd extraction
    const key = mm[3] ?? mm[2];
    const vExpr = mm[4].trim();
    const sv = jsResolveExpr(vExpr, bindings);
    if (sv !== null) out[key] = sv;
    else if (/^-?\d+(?:\.\d+)?$/.test(vExpr)) out[key] = parseFloat(vExpr);
    else if (vExpr === "true" || vExpr === "false") out[key] = vExpr === "true";
    // nested arrays/objects (update_plan payloads…) are irrelevant to the VFS
  }
  return out;
}

function parseToolProgram(vfs, src, resultText) {
  vfs._progDepth = (vfs._progDepth || 0) + 1;
  try {
    if (vfs._progDepth > 3) return;
    const failed = v4aFailedPaths(resultText);
    const mask = jsCodeMask(src);
    const bindings = new Map();
    const bRe = /\b(?:const|let|var)\s+([\w$]+)\s*=\s*/g;
    let m;
    while ((m = bRe.exec(src)) !== null) {
      if (!mask[m.index]) continue;
      const lit = jsResolveExpr(src.slice(bRe.lastIndex, jsStatementEnd(src, bRe.lastIndex)), bindings);
      if (lit !== null) bindings.set(m[1], lit);
    }
    const cRe = /\btools\s*\.\s*([\w$]+)\s*\(/g;
    while ((m = cRe.exec(src)) !== null) {
      if (!mask[m.index]) continue;
      const call = jsSliceCall(src, cRe.lastIndex - 1);
      if (!call) continue;
      cRe.lastIndex = call.end + 1; // never rescan inside consumed args
      const raw = call.arg.trim();
      let argv = parseJsObjectArg(raw, bindings);
      if (argv === null) argv = jsResolveExpr(raw, bindings);
      if (argv === null) { vfs.report.push("tool-program: unresolved args for tools." + m[1]); continue; }
      // real run rejected this inner patch — it threw, so neither the patch
      // nor anything after it in this program ever executed. When the error
      // names no path (codex "invalid hunk" style), the first patch call wins.
      if (typeof argv === "string" && argv.trimStart().startsWith("*** Begin Patch")
          && resultText && V4A_FAIL_RE.test(resultText)
          && (!failed.size || patchTargets(argv).some(t => [...failed].some(fp => samePath(t, fp))))) {
        vfs.report.push("tool-program: patch + rest of program skipped (real run rejected patch)");
        return;
      }
      applyToolCall(vfs, m[1], argv);
    }
  } finally { vfs._progDepth--; }
}

/* ---- codex V4A patches (`*** Begin Patch`) ---- */

function parseV4APatch(vfs, text) {
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  const nextHeader = j => j < lines.length && lines[j].startsWith("*** ");
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\*\*\*\s+Add File:\s*(.+)$/))) {
      const path = m[1].trim(); i++;
      const buf = [];
      while (i < lines.length && !nextHeader(i)) { buf.push(lines[i].startsWith("+") ? lines[i].slice(1) : lines[i]); i++; }
      while (buf.length && buf[buf.length - 1] === "") buf.pop();
      opWrite(vfs, path, buf.join("\n") + "\n", "exact", "apply_patch add");
      continue;
    }
    if ((m = line.match(/^\*\*\*\s+Delete File:\s*(.+)$/))) { opDelete(vfs, m[1].trim()); i++; continue; }
    if ((m = line.match(/^\*\*\*\s+Update File:\s*(.+)$/))) {
      const path = m[1].trim(); i++;
      let moveTo = null;
      if (i < lines.length && (m = lines[i].match(/^\*\*\*\s+Move to:\s*(.+)$/))) { moveTo = m[1].trim(); i++; }
      const hunks = []; let cur = null;
      while (i < lines.length && !nextHeader(i)) {
        const l = lines[i];
        if (l.startsWith("@@")) { if (cur && cur.length) hunks.push(cur); cur = []; i++; continue; }
        if (!cur) cur = [];
        if (l.startsWith("+")) cur.push({ tag: "+", text: l.slice(1) });
        else if (l.startsWith("-")) cur.push({ tag: "-", text: l.slice(1) });
        else if (l.startsWith(" ")) cur.push({ tag: " ", text: l.slice(1) });
        else if (l === "") cur.push({ tag: " ", text: "" });
        else cur.push({ tag: " ", text: l }); // tolerant: unprefixed context
        i++;
      }
      if (cur && cur.length) hunks.push(cur);
      applyHunkOps(vfs, path, hunks, "apply_patch");
      if (moveTo) opRename(vfs, path, moveTo);
      continue;
    }
    i++;
  }
}

function applyHunkOps(vfs, path, hunks, note) {
  const p = resolvePath(vfs, canonPath(path, vfs.cwd)); if (p === null) return;
  const f = vfsFile(vfs, p);
  f.ops++;
  if (f.content === null) { degrade(f, "unknown"); f.misses += hunks.length; vfsLog(f, note + " on unknown base"); return; }
  const r = applyHunks(f.content, hunks);
  f.content = r.content;
  f.misses += r.misses;
  if (r.rejected) { degrade(f, "approx"); vfs.report.push(note + ": patch rejected, " + r.misses + "/" + hunks.length + " hunks unmatched on " + p); }
  else if (r.fuzzy) degrade(f, "approx");
  vfsLog(f, note + " " + hunks.length + " hunks" + (r.rejected ? " (rejected: " + r.misses + " miss)" : ""));
}

/* ---- unified diffs (git apply / patch heredocs) ---- */

function parseUnifiedDiff(vfs, text) {
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].startsWith("--- ")) { i++; continue; }
    const from = lines[i].slice(4).trim().replace(/^[ab]\//, "");
    i++;
    if (i >= lines.length || !lines[i].startsWith("+++ ")) continue;
    const to = lines[i].slice(4).trim().replace(/^[ab]\//, "");
    i++;
    if (to === "/dev/null") { opDelete(vfs, from); continue; }
    const isNew = from === "/dev/null";
    const hunks = []; let cur = null; const addBuf = [];
    while (i < lines.length && !lines[i].startsWith("--- ") && !lines[i].startsWith("diff ")) {
      const l = lines[i];
      if (l.startsWith("@@")) { if (cur && cur.length) hunks.push(cur); cur = []; i++; continue; }
      if (!cur) { i++; continue; }
      if (l.startsWith("+")) { cur.push({ tag: "+", text: l.slice(1) }); addBuf.push(l.slice(1)); }
      else if (l.startsWith("-")) cur.push({ tag: "-", text: l.slice(1) });
      else if (l.startsWith(" ") || l === "") cur.push({ tag: " ", text: l.slice(1) });
      else if (l.startsWith("\\")) { /* \ No newline */ }
      else break;
      i++;
    }
    if (cur && cur.length) hunks.push(cur);
    if (isNew) opWrite(vfs, to, addBuf.join("\n") + "\n", "exact", "unidiff new");
    else applyHunkOps(vfs, to, hunks, "unidiff");
  }
}

/* ---- shell command main pipeline ---- */

function runShellCommand(vfs, rawCmd, depth) {
  depth = depth || 0;
  if (depth > 3 || typeof rawCmd !== "string" || !rawCmd) return;
  const cmd = unwrapShell(rawCmd);
  const { docs, shellLines } = splitHeredocs(cmd);
  const execScript = (kind, body) => {
    if (kind === "python") parsePythonScript(vfs, body);
    else if (kind === "node") parseNodeScript(vfs, body);
  };
  for (const doc of docs) {
    const cls = classifyHeredoc(doc.intro);
    switch (cls.kind) {
      case "file": {
        const fidelity = doc.quoted || !/[$`]/.test(doc.body) ? "exact" : "approx";
        if (cls.append) opAppend(vfs, cls.target, doc.body, fidelity, "heredoc>>");
        else opWrite(vfs, cls.target, doc.body, fidelity, "heredoc");
        break;
      }
      case "python": parsePythonScript(vfs, doc.body); if (cls.target) opStale(vfs, cls.target, "py stdout > file"); break;
      case "node": parseNodeScript(vfs, doc.body); if (cls.target) opStale(vfs, cls.target, "node stdout > file"); break;
      case "apply_patch": parseV4APatch(vfs, doc.body); break;
      case "unidiff": parseUnifiedDiff(vfs, doc.body); break;
      case "shell": runShellCommand(vfs, doc.body, depth + 1); break;
      default: break; // mysql, sqlite, unknown consumers — body already isolated
    }
  }
  for (const line of shellLines) scanShellLine(vfs, line, execScript);
}

/* ---- tool call iteration & routing ---- */

/* A tool call's payload sits under .function (OpenAI chat), under .custom
   (Responses-API freeform / codex `custom_tool_call`), or inline on the entry. */
function toolCallFn(tc) { return (tc && (tc.function || tc.custom)) || tc; }

function toolCallId(tc) {
  const fn = toolCallFn(tc);
  return tc.id ?? tc.tool_call_id ?? tc.call_id ?? fn.call_id ?? null;
}

function* iterToolCalls(m) {
  if (Array.isArray(m.tool_calls))
    for (const tc of m.tool_calls) {
      const fn = toolCallFn(tc);
      yield { name: fn.name || tc.name || "?", args: fn.arguments ?? fn.input, id: toolCallId(tc) };
    }
  if (m.function_call && m.function_call.name)
    yield { name: m.function_call.name, args: m.function_call.arguments, id: null };
  if (Array.isArray(m.content))
    for (const part of m.content)
      if (part && part.type === "tool_use") yield { name: part.name || "?", args: part.input, id: part.id ?? null };
}

const WRITE_PATH_KEYS = ["file_path", "path", "filename", "file", "target_file", "filePath", "targetFile", "fileName"];
const WRITE_BODY_KEYS = ["content", "file_text", "text", "code_edit", "contents", "new_str_content"];

function pickKey(a, keys) { for (const k of keys) if (typeof a[k] === "string") return a[k]; return null; }

/* Did the real run reject this patch? Mirror it: a rejected patch changed
   nothing on the real filesystem, so applying it here would drift. */
const V4A_FAIL_RE = /apply_patch verification failed|Failed to find expected lines|patch does not apply|invalid patch/i;

function v4aFailedPaths(resultText) {
  const out = new Set();
  if (!resultText) return out;
  for (const m of String(resultText).matchAll(/Failed to find expected lines in ([^\s:]+)/g)) out.add(m[1]);
  return out;
}

function patchTargets(patchText) {
  const out = [];
  for (const m of String(patchText).matchAll(/^\*\*\*\s+(?:Update|Add|Delete) File:\s*(.+)$/gm)) out.push(m[1].trim());
  return out;
}

function samePath(a, b) {
  if (!a || !b) return false;
  const ca = canonPath(a, ""), cb = canonPath(b, "");
  return ca === cb || ca.split("/").pop() === cb.split("/").pop();
}

function applyToolCall(vfs, name, rawArgs, resultText) {
  let a = rawArgs;
  const nm = String(name || "").toLowerCase();
  if (typeof a === "string") {
    const s = a.trim();
    if (s.startsWith("*** Begin Patch")) {
      if (resultText && V4A_FAIL_RE.test(resultText)) {
        vfs.report.push("apply_patch skipped — real run rejected it (" + patchTargets(s).join(", ") + ")");
        return;
      }
      parseV4APatch(vfs, s); return;
    }
    try { a = JSON.parse(a); } // JSON-string args: normal object path below
    catch {
      // not JSON: a JS orchestration program, a loose patch, or a raw command.
      // Program detection runs on non-JSON only, so command payloads that
      // merely mention "tools.x(" inside JSON stay on the object path.
      if (/\btools\s*\.\s*[\w$]+\s*\(/.test(s)) { parseToolProgram(vfs, s, resultText); return; }
      if (s.includes("*** Begin Patch")) { parseV4APatch(vfs, s); return; }
      if (/bash|shell|exec|terminal|command|cmd/.test(nm)) runShellCommand(vfs, a);
      return;
    }
  }
  if (a === null || typeof a !== "object") return;

  // apply_patch as {input|patch: "*** Begin Patch…"}
  const patchStr = typeof a.input === "string" && a.input.includes("*** Begin Patch") ? a.input
    : typeof a.patch === "string" && a.patch.includes("*** Begin Patch") ? a.patch : null;
  if (patchStr) {
    if (resultText && V4A_FAIL_RE.test(resultText))
      vfs.report.push("apply_patch skipped — real run rejected it (" + patchTargets(patchStr).join(", ") + ")");
    else parseV4APatch(vfs, patchStr);
    return;
  }

  // str_replace_editor family: {command: create|str_replace|insert, path, …}
  if (typeof a.command === "string" && typeof a.path === "string") {
    if (a.command === "create" && typeof a.file_text === "string") { opWrite(vfs, a.path, a.file_text, "exact", name); return; }
    if (a.command === "str_replace" && typeof a.old_str === "string") { opEditLiteral(vfs, a.path, a.old_str, a.new_str ?? "", false, name); return; }
    if (a.command === "insert" && typeof a.new_str === "string") {
      const p = resolvePath(vfs, canonPath(a.path, vfs.cwd));
      const f = p && vfs.files.get(p);
      if (f && f.content !== null) {
        const lines = f.content.split("\n");
        const at = Math.max(0, Math.min(lines.length, (a.insert_line | 0)));
        lines.splice(at, 0, ...a.new_str.split("\n"));
        f.content = lines.join("\n"); f.ops++; vfsLog(f, "insert@" + at);
      } else if (p) opStale(vfs, p, "insert on unknown base");
      return;
    }
  }

  if (/notebook/.test(nm)) return; // NotebookEdit et al: not web artifacts

  const path = pickKey(a, WRITE_PATH_KEYS);
  const body = pickKey(a, WRITE_BODY_KEYS);
  const isReadTool = /^(read|view|open|cat|get)/.test(nm);

  // Some harnesses (kilo-code family, skill/task family) spell these camelCase — filePath /
  // oldString / newString / replaceAll. Missing the aliases is silent: every edit no-ops and the
  // trajectory looks like "the agent never wrote a file" (798 records in one 4k corpus).
  if (path && Array.isArray(a.edits)) { // MultiEdit
    for (const e of a.edits)
      if (e && typeof (e.old_string ?? e.oldString) === "string")
        opEditLiteral(vfs, path, e.old_string ?? e.oldString, e.new_string ?? e.newString ?? "",
                      !!(e.replace_all ?? e.replaceAll), "MultiEdit");
    return;
  }
  if (path && typeof (a.old_string ?? a.old_str ?? a.oldString) === "string") { // Edit
    opEditLiteral(vfs, path, a.old_string ?? a.old_str ?? a.oldString,
                  a.new_string ?? a.new_str ?? a.newString ?? "", !!(a.replace_all ?? a.replaceAll), name);
    return;
  }
  // StrReplaceFile family: {path, edit: {old, new}} or {path, edit: [{old, new}, …]}. The `edit`
  // key is singular and the inner keys are bare old/new, so neither the MultiEdit nor the Edit
  // branch above sees it — every replacement silently no-ops and the page stays a skeleton.
  if (path && a.edit && typeof a.edit === "object") {
    for (const e of (Array.isArray(a.edit) ? a.edit : [a.edit])) {
      const o = e && (e.old ?? e.old_string ?? e.oldString);
      if (typeof o === "string")
        opEditLiteral(vfs, path, o, e.new ?? e.new_string ?? e.newString ?? "",
                      !!(e.replace_all ?? e.replaceAll), name);
    }
    return;
  }
  // mode:"append" means append, not overwrite. Treating it as a write keeps only the LAST chunk —
  // a page assembled as doctype + N appended sections collapses to its final <script> fragment,
  // which still renders (badly), so this corrupts scores instead of merely losing records.
  if (path && body !== null && !isReadTool && (a.mode === "append" || a.mode === "a")) {
    opAppend(vfs, path, body, null, name); return;
  }
  if (path && body !== null && !isReadTool) { opWrite(vfs, path, body, "exact", name); return; }

  // command-bearing tools (exec_command/Bash/shell/run/…)
  const cmd = extractCommand(a);
  if (cmd) {
    const wd = a.workdir ?? a.cwd;
    if (typeof wd === "string" && wd.startsWith("/")) vfs.cwd = wd;
    runShellCommand(vfs, cmd);
  }
}

function toolResultText(m) {
  const c = m.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map(p => typeof p === "string" ? p : (p && p.text) || "").join("\n");
  return "";
}

/* id linking a tool result back to its call: .tool_call_id (OpenAI chat) or
   .call_id (Responses-API {custom,function}_call_output) */
function msgCallId(m) { return m.tool_call_id ?? m.call_id ?? m.id ?? null; }

function buildVfs(msgs) {
  const vfs = vfsNew();
  const byId = new Map();
  for (const m of msgs)
    if (m.role === "tool" && msgCallId(m)) byId.set(msgCallId(m), toolResultText(m));
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role !== "assistant") continue;
    // positional fallback for results when tool_call_id is absent
    const seq = [];
    for (let k = i + 1; k < msgs.length && msgs[k].role === "tool"; k++) seq.push(msgs[k]);
    let si = 0;
    for (const { name, args, id } of iterToolCalls(m)) {
      const rt = (id != null && byId.has(id)) ? byId.get(id)
        : (si < seq.length ? toolResultText(seq[si]) : "");
      si++;
      try { applyToolCall(vfs, name, args, rt); }
      catch (e) { vfs.report.push("parser error in " + name + ": " + (e.message || e)); }
    }
  }
  // prune noise: directories accidentally registered, size-0 unknowns with no ops
  for (const [p, f] of [...vfs.files]) {
    if (f.content === null && f.status === "unknown" && !f.ops) vfs.files.delete(p);
  }
  return vfs;
}

/* ---- entry pick & preview composition ---- */

function pickEntry(vfs) {
  const all = [...vfs.files.keys()].filter(p =>
    /\.(html?|svg)$/i.test(p) && typeof vfs.files.get(p).content === "string");
  if (!all.length) return null;
  const rank = p => /(^|\/)index\.html?$/i.test(p) ? 0 : /\.html?$/i.test(p) ? 1 : 2;
  const depth = p => p.split("/").length;
  // products sometimes live under /tmp (e.g. /tmp/out/<task>/index.html);
  // tmp paths are only deprioritized, not banned
  const main = all.filter(p => !isTmpPath(p));
  const cand = main.length ? main : all;
  cand.sort((x, y) => rank(x) - rank(y) || depth(x) - depth(y)
    || vfs.files.get(y).content.length - vfs.files.get(x).content.length);
  return cand[0];
}

