/**
 * app8.nextaura.fit — Lumen DAW (IBM CE) + agent-index proxy to training lab.
 * Agent/scrape paths go to app8.nextaura.us; everything else stays on CE origin.
 */
const TRAINING_ORIGIN = "https://app8.nextaura.us";
const TRAINING_CANONICAL = "https://app8.nextaura.us";
const LUMEN_ORIGIN =
  "https://app8-nextaura-fit.284w7l87aq94.us-south.codeengine.appdomain.cloud";

const AGENT_EXACT = new Set([
  "/agent.json",
  "/agent.txt",
  "/llms.txt",
  "/robots.txt",
  "/sitemap.xml",
  "/.well-known/agent.json",
  "/api/agent/index",
]);

const API_PREFIX = "/api/";

function isTrainingPath(pathname) {
  if (AGENT_EXACT.has(pathname)) return true;
  if (pathname.startsWith(API_PREFIX)) return true;
  return false;
}

async function proxyTraining(request) {
  const incoming = new URL(request.url);
  const target = new URL(incoming.pathname + incoming.search, TRAINING_ORIGIN);
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.set("X-Forwarded-Host", incoming.host);
  headers.set("X-Forwarded-Proto", incoming.protocol.replace(":", ""));
  headers.set("X-Canonical-Host", "app8.nextaura.us");
  const res = await fetch(target.toString(), {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "follow",
  });
  const out = new Headers(res.headers);
  out.set("x-proxied-by", "nextaura-app8-fit-agent-proxy");
  out.set("link", `<${TRAINING_CANONICAL}${incoming.pathname}>; rel="canonical"`);
  return new Response(res.body, { status: res.status, headers: out });
}

async function proxyLumen(request) {
  const incoming = new URL(request.url);
  const target = new URL(LUMEN_ORIGIN);
  target.pathname = incoming.pathname;
  target.search = incoming.search;
  const headers = new Headers(request.headers);
  headers.delete("host");
  const res = await fetch(target.toString(), {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "follow",
  });
  const out = new Headers(res.headers);
  out.set("x-proxied-by", "nextaura-app8");
  return new Response(res.body, { status: res.status, headers: out });
}

export default {
  async fetch(request) {
    const pathname = new URL(request.url).pathname;
    if (isTrainingPath(pathname)) {
      return proxyTraining(request);
    }
    return proxyLumen(request);
  },
};
