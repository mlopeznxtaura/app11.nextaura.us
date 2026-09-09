const ORIGIN = "http://150-239-227-60.sslip.io";

export default {
  async fetch(request) {
    const incoming = new URL(request.url);
    const outbound = new URL(ORIGIN);
    outbound.pathname = incoming.pathname;
    outbound.search = incoming.search;
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.set("X-Forwarded-Host", incoming.host);
    headers.set("X-Forwarded-Proto", incoming.protocol.replace(":", ""));
    const res = await fetch(outbound.toString(), {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      redirect: "follow",
    });
    const out = new Headers(res.headers);
    out.set("x-proxied-by", "nextaura-app9-us");
    return new Response(res.body, { status: res.status, headers: out });
  },
};
