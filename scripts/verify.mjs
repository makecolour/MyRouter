#!/usr/bin/env node
// Verify a MyRouter / OpenAI-compatible chat endpoint in BOTH modes.
//
//   node scripts/verify.mjs <base_url> <api_key> [model]
//
// Examples:
//   # Through 9Router
//   node scripts/verify.mjs https://9router.montserrat.id.vn/v1 sk-...key mr/gemini-3-flash
//   # Directly against the sidecar (byte-clean both ways)
//   node scripts/verify.mjs http://localhost:8000/v1 sk-...key gemini-3-flash
//
// Non-stream is parsed defensively (tolerates 9Router appending a stray
// "data: [DONE]" after the JSON). Streaming is parsed as real OpenAI SSE.
// Needs Node 18+ (global fetch). No dependencies.

const [baseUrl, apiKey, model = "gemini-3-flash"] = process.argv.slice(2);

if (!baseUrl || !apiKey) {
  console.error("Usage: node scripts/verify.mjs <base_url> <api_key> [model]");
  console.error("  <base_url> ends in /v1 (e.g. http://localhost:8000/v1)");
  process.exit(2);
}

const url = baseUrl.replace(/\/+$/, "") + "/chat/completions";
const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${apiKey}`,
};
const prompt = "Reply with exactly: streaming works";

let failures = 0;
const pass = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  " + detail : ""}`);
};

// ---- Non-stream (defensive parse) ----
async function testNonStream() {
  console.log("\n=== NON-STREAM ===");
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
  });
  const text = await res.text();
  // 9Router may append "data: [DONE]" after the JSON — cut it off.
  const jsonPart = text.split("data: [DONE]")[0].trim();
  let data;
  try {
    data = JSON.parse(jsonPart);
  } catch (e) {
    pass("non-stream parses", false, `HTTP ${res.status} — ${String(e)}`);
    console.log("  raw (first 300):", text.slice(0, 300));
    return;
  }
  const content = data.choices?.[0]?.message?.content ?? "";
  const hadTrailingDone = text.includes("data: [DONE]");
  pass("non-stream returns content", res.ok && !!content);
  pass("usage present", !!data.usage, JSON.stringify(data.usage || {}));
  if (hadTrailingDone) {
    console.log("  note: upstream appended a trailing 'data: [DONE]' — defensive parse handled it.");
  }
  console.log("  content:", JSON.stringify(content).slice(0, 200));
}

// ---- Streaming (real OpenAI SSE) ----
async function testStream() {
  console.log("\n=== STREAMING ===");
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      stream: true,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const ct = res.headers.get("content-type") || "";
  if (!res.body) {
    pass("stream has a body", false, `HTTP ${res.status}, content-type ${ct}`);
    console.log("  raw:", (await res.text()).slice(0, 300));
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  let chunks = 0;
  let usage = null;
  const t0 = Date.now();
  let firstAt = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep partial line
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const payload = t.slice(5).trim();
      if (payload === "[DONE]") continue;
      let obj;
      try {
        obj = JSON.parse(payload);
      } catch {
        continue;
      }
      if (obj.usage) usage = obj.usage;
      const delta = obj.choices?.[0]?.delta?.content;
      if (delta) {
        if (firstAt === null) firstAt = Date.now() - t0;
        content += delta;
        chunks++;
      }
    }
  }

  pass("stream returns content", !!content);
  pass("usage present on stream", !!usage, JSON.stringify(usage || {}));
  console.log(
    `  content chunks: ${chunks}` +
      (firstAt !== null ? `  |  first token at ${firstAt}ms` : "")
  );
  console.log("  content:", JSON.stringify(content).slice(0, 200));
}

(async () => {
  console.log(`endpoint: ${url}\nmodel:    ${model}`);
  try {
    await testNonStream();
  } catch (e) {
    pass("non-stream request", false, String(e));
  }
  try {
    await testStream();
  } catch (e) {
    pass("stream request", false, String(e));
  }
  console.log(failures ? `\n${failures} check(s) FAILED` : "\nALL CHECKS PASSED");
  process.exit(failures ? 1 : 0);
})();
