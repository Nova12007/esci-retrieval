// Open-loop load test for /search (k6).
//
//   docker run --rm -i -v "$PWD/loadtest:/lt" -e BASE_URL=https://... \
//       grafana/k6 run /lt/search.js
//
// Arrival rate is fixed, not paced by responses: a closed-loop test slows
// down when the server does and so hides the queueing a p99 should show.
// Queries are all 22,458 unique test-fold queries, shuffled once and walked
// in order, never repeated within a run: every request is a cache miss, the
// worst case. (Sampling 10k requests from 2,000 queries gave an 81% hit rate.)

import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import exec from "k6/execution";

const BASE = __ENV.BASE_URL || "http://host.docker.internal:7860";
const PEAK = Number(__ENV.PEAK_RPS || 50);
const queries = new SharedArray("queries", () => JSON.parse(open("./queries.json")));

export const options = {
  scenarios: {
    search: {
      executor: "ramping-arrival-rate",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 50,
      maxVUs: 300,
      stages: [
        { target: PEAK, duration: "1m" }, // ramp
        { target: PEAK, duration: "3m" }, // hold at peak
      ],
    },
  },
  summaryTrendStats: ["avg", "p(50)", "p(95)", "p(99)", "max"],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{phase:peak}": ["p(99)<100"],
  },
};

export default function () {
  const q = queries[exec.scenario.iterationInTest % queries.length];
  // Tag requests in the hold phase so the threshold judges steady state, not the ramp.
  const peak = Date.now() - exec.scenario.startTime > 60_000;
  const res = http.get(`${BASE}/search?k=10&q=${encodeURIComponent(q)}`, {
    tags: { phase: peak ? "peak" : "ramp" },
  });
  check(res, { "status 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data),
    "/lt/summary.json": JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  const d = data.metrics.http_req_duration.values;
  const peak = data.metrics["http_req_duration{phase:peak}"];
  const fmt = (v) => `p50 ${v["p(50)"].toFixed(1)} ms, p95 ${v["p(95)"].toFixed(1)} ms, p99 ${v["p(99)"].toFixed(1)} ms`;
  return [
    `requests: ${data.metrics.http_reqs.values.count}, failed: ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%`,
    `all:  ${fmt(d)}`,
    peak ? `peak: ${fmt(peak.values)}` : "peak: (no samples)",
    "",
  ].join("\n");
}
