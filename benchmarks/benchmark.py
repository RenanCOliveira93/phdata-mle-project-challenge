"""Latency/throughput benchmark for POST /predict (standard library only).

Runs against a live API, e.g. the Docker container:
    docker run -d -p 8000:8000 --name housing-api mle-project-challenge-2026
    python benchmarks/benchmark.py http://localhost:8000 \\
        src/data/future_unseen_examples.csv results.json

Scenarios, both built from the rows of future_unseen_examples.csv:
    A "complete":   the 8 API fields of every row, unchanged.
    B "incomplete": the same rows with 1-7 house fields set to null
                    (fixed seed, so every run sends identical requests).

For each scenario, after 20 warm-up requests on a keep-alive connection:
    sequential  - 3 passes over the 100 rows (300 requests)
    workers=1   - 500 requests from one client thread
    workers=10  - 500 requests from 10 concurrent client threads
Non-200 responses and connection errors are counted, not raised. Results
(including the predictions of the first sequential pass) go to the JSON file.
"""
import csv
import http.client
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

HOUSE_FIELDS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                "sqft_above", "sqft_basement"]


def load_scenarios(csv_path):
    complete = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            payload = {k: float(row[k]) for k in HOUSE_FIELDS}
            payload["bedrooms"] = int(payload["bedrooms"])
            payload["zipcode"] = row["zipcode"]
            complete.append(payload)
    rng = random.Random(7)
    incomplete = []
    for payload in complete:
        missing = rng.sample(HOUSE_FIELDS, rng.randint(1, len(HOUSE_FIELDS)))
        incomplete.append({k: (None if k in missing else v) for k, v in payload.items()})
    return {"complete": complete, "incomplete": incomplete}


class Client:
    def __init__(self, url):
        self.url = url
        self.conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)

    def post(self, payload):
        """Return (latency_ms, status, body); status None on connection error."""
        body = json.dumps(payload)
        start = time.perf_counter()
        try:
            self.conn.request("POST", "/predict", body, {"Content-Type": "application/json"})
            response = self.conn.getresponse()
            data = response.read()
            status = response.status
        except (OSError, http.client.HTTPException):
            self.conn.close()
            self.conn = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=30)
            return (time.perf_counter() - start) * 1000, None, None
        return (time.perf_counter() - start) * 1000, status, data


def summarize(latencies, errors, wall_seconds=None):
    xs = sorted(latencies)
    pct = lambda q: xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    result = {"n": len(xs), "errors": errors, "mean_ms": statistics.mean(xs),
              "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
              "min_ms": xs[0], "max_ms": xs[-1]}
    if wall_seconds:
        result["rps"] = len(xs) / wall_seconds
    return result


def run(client_factory, rows, n, start=0):
    client = client_factory()
    latencies, errors, results = [], 0, []
    for i in range(n):
        latency, status, data = client.post(rows[(start + i) % len(rows)])
        latencies.append(latency)
        if status != 200:
            errors += 1
            results.append(None)
        else:
            results.append(json.loads(data)["predicted_price"])
    return latencies, errors, results


def benchmark(url, rows):
    factory = lambda: Client(url)
    out = {}
    first_latency, first_status, _ = factory().post(rows[0])
    out["first_request"] = {"ms": first_latency, "status": first_status}
    run(factory, rows, 20)  # warm-up

    latencies, errors, results = run(factory, rows, 3 * len(rows))
    out["sequential"] = summarize(latencies, errors)
    out["predictions"] = results[:len(rows)]

    for workers in (1, 10):
        total = 500
        start = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(lambda w: run(factory, rows, total // workers, w * 7),
                                  range(workers)))
        wall = time.perf_counter() - start
        out[f"workers_{workers}"] = summarize(
            [x for p in parts for x in p[0]], sum(p[1] for p in parts), wall)
    return out


def main():
    url, csv_path, out_path = urlparse(sys.argv[1]), sys.argv[2], sys.argv[3]
    results = {name: benchmark(url, rows) for name, rows in load_scenarios(csv_path).items()}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    for name, r in results.items():
        print(f"[{name}] first request: {r['first_request']['ms']:.2f}ms "
              f"(HTTP {r['first_request']['status']})")
        for key in ("sequential", "workers_1", "workers_10"):
            s = r[key]
            rps = f"  rps={s['rps']:6.1f}" if "rps" in s else ""
            print(f"[{name}] {key:<10} n={s['n']} errors={s['errors']} mean={s['mean_ms']:.2f}ms "
                  f"p50={s['p50_ms']:.2f} p95={s['p95_ms']:.2f}{rps}")


if __name__ == "__main__":
    main()
