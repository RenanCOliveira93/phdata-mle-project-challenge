# Sound Realty House-Price API

A FastAPI service that estimates the sale price of a house in the Seattle area from seven property attributes and the zipcode. The zipcode is used to attach census demographics.

This submission keeps the provided model (`src/model/model.pkl`, a `RobustScaler` + `KNeighborsRegressor` pipeline) unchanged and adds:

1. **Missing-data handling:** any field except `zipcode` may be missing. Missing values are filled with the KNN imputation selected in `notebooks/imputation_experiment.ipynb`.
2. **Performance:** static resources are loaded once at startup instead of on every request. Warm latency for complete requests dropped by about 70% (measurements below).
3. **Code quality:** input validation, clear errors, logging, pinned dependencies, and a reorganized test suite with a non-regression check against the original API.

The model was **not retrained**. For complete requests, the API returns exactly the same predictions as the original implementation.

**Four kinds of evidence, kept separate below:**

| Question | Evidence | Where |
|---|---|---|
| Do complete requests behave exactly as before? | 100/100 predictions identical to the original API | [Tests](#tests) |
| How much does missing data hurt house-price predictions? | Held-out evaluation with 15% of house values masked: MAE +3.0% with KNN vs +7.6% with mean imputation | [Held-out evaluation](#held-out-evaluation-missing-data) |
| How well are the missing values reconstructed? | Per-feature error of KNN vs the mean on the masked cells | [Held-out evaluation](#held-out-evaluation-missing-data) |
| Is the API faster? | Interleaved Docker benchmark: about 70% lower latency, about 3.4× the throughput | [Benchmark](#benchmark) |

---

## Quick start (Docker)

Prerequisite: Docker. The model artifacts are already in the repository; no training step is needed.

```bash
docker build -t mle-project-challenge-2026 .
docker run -d -p 8000:8000 --name housing-api mle-project-challenge-2026
curl http://localhost:8000/health          # {"status":"healthy"}
```

- Interactive docs: <http://localhost:8000/docs>
- Logs: `docker logs housing-api`
- Log level: `-e LOG_LEVEL=DEBUG`. DEBUG also logs the full feature row sent to the model.
- Stop / remove: `docker stop housing-api && docker rm housing-api`
- If port 8000 is already in use, map another host port, e.g. `-p 18000:8000`.

<details>
<summary>Regenerating the model (not needed; retrains and overwrites the committed artifacts)</summary>

```bash
docker build -f src/model/Dockerfile -t create-model .
docker run --rm -v "$(pwd)/src/model:/app/model" create-model
```

This runs the root `create_model.py`. The copy in `src/model/` is an unused duplicate that was provided with the project.
</details>

---

## API

### `POST /predict`

| Field | Type | Required | Constraint |
|---|---|---|---|
| `zipcode` | string | **yes** | must be one of the 70 zipcodes in `zipcode_demographics.csv` |
| `bedrooms` | integer | no | ≥ 0 |
| `bathrooms` | number | no | ≥ 0 |
| `sqft_living` | number | no | > 0 |
| `sqft_lot` | number | no | > 0 |
| `floors` | number | no | > 0 |
| `sqft_above` | number | no | ≥ 0 |
| `sqft_basement` | number | no | ≥ 0 |

- **Missing fields:** a field is "missing" if it is `null` or left out.
- **Extra fields** (e.g. full listings with `waterfront`, `grade`, …) are ignored.
- **Bounds:** they are deliberately loose. Every row of the training data and of `future_unseen_examples.csv` passes them.

**Complete request**
```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "bedrooms": 3, "bathrooms": 2.5, "sqft_living": 2000, "sqft_lot": 5000,
  "floors": 2, "sqft_above": 1500, "sqft_basement": 500, "zipcode": "98125"}'
# {"predicted_price":518190.0}
```

**Request with missing data**
```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "bedrooms": 3, "bathrooms": null, "sqft_living": 2000, "sqft_lot": null,
  "floors": 2, "sqft_above": 1500, "sqft_basement": 500, "zipcode": "98125"}'
# {"predicted_price":606000.0}
```

The response has the same shape for every successful request: `{"predicted_price": <float>}`. It does not say which fields were imputed. That information goes to the logs instead, e.g. `INFO api.endpoints: Imputing ['bathrooms', 'sqft_lot'] for zipcode 98125`.

**Errors**

| Case | Status | Body |
|---|---|---|
| `zipcode` missing, `null` or not a string | 422 | FastAPI validation error |
| Unknown zipcode (e.g. `"00000"`) | 422 | `{"detail": "Unknown zipcode: 00000"}` |
| Out-of-range, non-numeric, non-integer `bedrooms` | 422 | FastAPI validation error naming the field |
| `NaN` / `Infinity` literals | 422 | `"Input should be a finite number"` (missing values must be `null` or omitted) |
| Malformed JSON body | 422 | FastAPI validation error |

Before these changes, an unknown zipcode or `Infinity` caused a 500 error, and any `null` was rejected.

### `GET /health`
Returns `{"status": "healthy"}`. All resources are loaded before the server starts accepting requests, so a healthy response means the model is ready.

---

## How it works

### Request flow

```
startup (once): load model.pkl, model_features.json, zipcode demographics -> dict,
                fit KNNImputer on the model's training rows
request:
  validate (Pydantic) ──► 422 on invalid input
  demographics[zipcode] ──► 422 if unknown
  any house field missing? ── yes ──► imputer.transform(7 house fields)
                          └─ no ───► skipped (identical path to the original API)
  one row: 7 house fields + 26 demographic columns, in model_features.json order
  NaN guard (defensive) ──► model.predict ──► {"predicted_price": ...}
```

Code layout:
- `src/main.py`: app, logging setup, 422 handler.
- `src/api/endpoints.py`: request schema, resource loading, `/predict`.
- `src/utils/loader.py`: loading functions and `fit_imputer`.

### KNN imputation

The approach follows the Data Science team's notebook:

- **Configuration:** `KNNImputer(n_neighbors=5, weights="distance")`, with the default `nan_euclidean` distance and **no scaling**, exactly as in the notebook.
- **Imputed features:** only the 7 house attributes the API receives.
  - Never used: `price` (the target), ids, dates and `zipcode`.
  - The 26 demographic columns aren't imputed. They are fully determined by the zipcode, which is required, so they always come from the lookup.
  - The notebook's other 10 columns (`grade`, `lat`, …) aren't sent to the API. At inference time they would always be missing.
- **Reference data:** the 7 house columns of `kc_house_data.csv`, limited to the **model's training split**. That's 16,209 rows, from the same `train_test_split(random_state=42)` call as `create_model.py`. The 5,404 held-out rows are never used as neighbors, so they stay clean for evaluation. The notebook used the full dataset; this is a deliberate, documented difference.
- **Fitting:** done once at startup, in under 1 ms, because `KNNImputer.fit` only stores the reference rows. There is no extra artifact to build or version.
- **Complete requests skip the imputer.** Running a complete row through it was also shown to return the row unchanged, so predictions match the original API exactly either way.
- **All 7 house fields missing:** this is accepted. KNNImputer falls back to the training-set column means, and a WARNING is logged. The prediction then mostly reflects the zipcode.

### Performance changes

I measured each step of the original handler before changing anything. Median cost per request:

| Step | Original cost per request | Now |
|---|---|---|
| `print(input_data)`, formatting a 33-column DataFrame | **5.98 ms** | removed (lazy `logger.debug`) |
| `read_csv` demographics, then filter + `concat` | 0.66 + 0.40 ms | dict lookup built once at startup |
| `pickle.load` model | 0.13 ms | loaded once at startup |
| `json.load` features | 0.007 ms | loaded once at startup |
| `model.predict` (brute-force KNN over 16,209 rows) | 2.70 ms | unchanged (part of the model) |

Other changes:
- `--reload`, a development file watcher, was removed from the container command.
- A `.dockerignore` keeps the local virtualenv out of the build context, which is now 4.4 MB.

---

## Tests

Everything runs in Docker:

```bash
make test-unit          # 41 unit tests (FastAPI TestClient) + coverage report in test-results/
make test-integration   # 4 tests against the real API container via docker-compose (uses host port 8000)
make test-all
```

| File | Covers |
|---|---|
| `test/unit/test_imputation.py` | **A. Imputation:** the notebook configuration; reference data is the model's training split; complete rows are unchanged and complete requests skip the imputer; only missing cells are filled, and they stay within the reference range; one, several, or all fields missing |
| `test/unit/test_prediction_regression.py` | **B. Non-regression:** all 100 `future_unseen` rows return exactly the original API's predictions; the model input has the right column order and no NaN; demographics lookup matches the CSV; results are deterministic |
| `test/unit/test_api_unit.py` | **C. Robustness:** health; missing, malformed and unknown zipcodes; invalid and non-finite numbers; malformed bodies; extra listing columns |
| `test/unit/test_performance.py` | **D. Performance:** the test fails if the model, features, CSVs or imputer are loaded or fitted during a request |
| `test/integration/test_api_integration.py` | The Docker container: health, complete and missing-data predictions, parity with the original API on all 100 rows |

**Results:** 41 unit tests and 4 integration tests pass. Line coverage is 97%. The uncovered lines are the defensive NaN guard and the `__main__` block.

The suite was checked with deliberately broken versions of the code. Each of these made at least one test fail:
- the model reloaded on every request;
- the imputer always called;
- the feature order scrambled;
- the unknown-zipcode check removed;
- `weights="uniform"`;
- the imputer fitted on the full dataset.

**About the parity reference:** `test/fixtures/baseline_predictions.json` holds the **original** API's predictions (commit `a3e1e36`) for `future_unseen_examples.csv`. That file has no prices, so these tests show *consistency with the original API*, **not accuracy**.

---

## Benchmark

**Method:** [`benchmarks/benchmark.py`](benchmarks/benchmark.py) uses only the Python standard library.

- **Images compared:** each image was built from its own repository state. The original (commit `a3e1e36`) used its own Dockerfile, which included `--reload` and unpinned dependencies (its build resolved fastapi 0.142.2 instead of the pinned 0.141.1).
- **Runs:** each image ran in its own new container on the same laptop (Docker Desktop, 14 CPUs, 8 GB), in **3 interleaved rounds**. All rounds are reported.
- **Requests:** the 100 rows of `future_unseen_examples.csv`, sent after 20 warm-up requests:
  - sequentially, 3 passes = 300 requests;
  - 500 requests from 1 client;
  - 500 requests from 10 concurrent clients.
- **Scenario B:** the same rows with 1–7 fields set to `null`, chosen with a fixed seed.

```bash
docker run -d -p 8000:8000 --name housing-api mle-project-challenge-2026
python benchmarks/benchmark.py http://localhost:8000 src/data/future_unseen_examples.csv results.json
```

**A. Complete requests:** mean of the 3 rounds [min–max]. There were 0 errors in every run for both images.

| Measure | Original | Final | Change |
|---|---|---|---|
| Sequential mean (n=300) | 11.63 ms [11.48–11.89] | 3.44 ms [3.36–3.54] | −70.4% |
| Sequential p50 / p95 | 11.67 / 12.87 ms | 3.65 / 4.14 ms | p95 −67.8% |
| 1 client, throughput (n=500) | 86.7 req/s | 292.2 req/s | ×3.37 |
| 10 clients, throughput (n=500) | 92.4 req/s | 326.1 req/s | ×3.53 |
| 10 clients, p50 / p95 | 107.8 / 115.0 ms | 30.2 / 35.0 ms | p95 −69.6% |
| Container start until `/health` responds | 0.555 s | 0.794 s | **+0.24 s (slower)** |

**B. Requests needing imputation:** the original API rejects all of them (300/300 and 500/500 got a 422), so there is no latency to compare. The final API serves them with 0 errors:

| Measure | Final |
|---|---|
| Sequential mean / p50 / p95 | 5.68 / 5.69 / 6.86 ms |
| 1 client / 10 clients throughput | 176.2 / 184.1 req/s |

The extra ~2.2 ms over complete requests is **consistent with** imputation being the main additional cost: in a separate in-process measurement on the same payloads, `imputer.transform` took 2.33 ms (median). The two numbers come from different experiments.

These numbers come from Docker Desktop on a laptop, so they are only meaningful for comparing the two versions, not as a capacity estimate. The server is still a single process with a blocking handler. With 10 clients, throughput is only about 1.1× that of 1 client, because requests queue.

---

## Held-out evaluation (missing data)

[`notebooks/final_evaluation.ipynb`](notebooks/final_evaluation.ipynb) is technical evidence, not application code. It runs the **production** `fit_imputer` and the **unchanged** model.

**Design:**
- **Data:** the model's own split, `train_test_split(random_state=42)`, as in `create_model.py`.
- **Fitting:** both imputers are fitted on the 16,209 training rows only.
- **Masking:** 15% of each house feature in the 5,404 held-out rows is masked completely at random (MCAR). That's 810 cells per feature, 14.99% of all cells, with `RandomState(42)`, following the Data Science notebook's protocol. 68.2% of rows have at least one masked feature.
- **Same cells for both methods:** KNN and the mean baseline fill exactly the same masked cells.
- **What the imputers see:** never `price`, ids, dates, zipcode or demographics.
- **Differences from the notebook:** only the 7 API features are masked (not 17), and masking is applied to held-out rows only, with no `fit_transform` on the evaluation data.
- **Check:** imputing rows one at a time, as the API does, gives exactly the batch values.

**House-price prediction error, all 5,404 held-out rows (USD):**

| Inputs | MAE | RMSE | R² | MAE vs complete | RMSE vs complete |
|---|---|---|---|---|---|
| Complete | 102,067 | 201,680 | 0.7281 | – | – |
| KNN-imputed (production) | 105,164 | 208,814 | 0.7085 | +3,097 (+3.0%) | +7,134 (+3.5%) |
| Mean-imputed (baseline only, not in the API) | 109,849 | 212,495 | 0.6981 | +7,781 (+7.6%) | +10,815 (+5.4%) |

- **KNN vs mean:** MAE −4,685 (−4.3%), RMSE −3,682 (−1.7%), R² +0.0104.
- **Masking seeds 0–4:** KNN had the lower MAE for every seed (by 4,371 to 6,928).
- **Only the 3,687 rows with at least one masked feature:** MAE 104,761 complete, 109,300 KNN, 116,167 mean.

**Reconstruction error on the masked cells**, in each feature's own units. The raw values can't be compared across features. The notebook also shows MAE divided by the training std.

| Feature | KNN MAE | Mean MAE | KNN RMSE | Mean RMSE |
|---|---|---|---|---|
| bedrooms | 0.579 | 0.723 | 0.818 | 0.895 |
| bathrooms | 0.368 | 0.619 | 0.524 | 0.778 |
| sqft_living | 170.5 | 706.3 | 412.1 | 920.1 |
| sqft_lot | **17,973** | **15,293** | **76,921** | **71,059** |
| floors | 0.199 | 0.490 | 0.334 | 0.537 |
| sqft_above | 154.0 | 649.4 | 377.3 | 819.1 |
| sqft_basement | 110.4 | 364.3 | 251.1 | 446.4 |

KNN reconstructs 6 of the 7 features better than the mean. For `sqft_lot` the mean is better, as it also was in the Data Science notebook.

**Limits of these numbers:**
- The masking is synthetic, at one rate, on one split.
- These are error metrics on historical data, not a claim of accuracy for future listings. `future_unseen_examples.csv` has no prices.

To re-run (about 35 s, Docker):

```bash
docker run --rm -v "$PWD:/work" -w /work/notebooks python:3.10-slim sh -c \
  "pip install -q -r ../requirements.txt -r requirements-notebook.txt && \
   jupyter nbconvert --to notebook --execute --inplace final_evaluation.ipynb"
```

---

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Imputation on the 7 API house fields, no scaling | Faithful to the notebook's configuration | Among the 7 raw features, `sqft_lot` accounts for ~99.9% of the variance, so it can strongly influence the unscaled distance whenever it is present |
| Imputer reference = model training split | Held-out rows stay unused for later evaluation | Small, documented difference from the notebook (which used the full dataset) |
| Fit the imputer at startup instead of a pickled artifact | No build step; fitting takes under 1 ms | Startup reads the 2.5 MB sales CSV (+0.24 s to healthy) |
| Load resources at import, not in a lifespan hook | The test client in `conftest.py` doesn't trigger lifespan events | Replacing artifacts on disk now needs a restart; importing the module has side effects |
| Response contract unchanged | Responses for complete requests stay byte-identical | Clients aren't told which values were imputed (only the logs are) |
| Accept requests with all 7 house fields missing | The brief allows any field except zipcode to be missing | The prediction is mostly driven by zipcode demographics |
| Unknown zipcode → 422 | It's invalid input for what the model covers; previously a 500 | – |
| Custom 422 handler | FastAPI's default handler crashes (500) when echoing `NaN`/`Infinity` inputs | Identical body to the default, except that unencodable input echoes are dropped |
| Pinned runtime and test dependencies | `model.pkl` was pickled with scikit-learn 1.7.2; imputation results were verified only with that version, and other versions may differ | Upgrades must be deliberate. Indirect dependencies are not locked |

## Known limitations

- **The missing-data evaluation is synthetic:** missing values were masked completely at random, at one rate (15% per feature), on one held-out split. Real listings probably miss values systematically (e.g. a blank basement field), and that isn't covered. See the [held-out evaluation](#held-out-evaluation-missing-data).
- **KNN is worse than the mean at reconstructing `sqft_lot`.** It's better for the other 6 features.
- **Imputed values aren't constrained:** they can be fractional (e.g. 2.8 bedrooms) and need not satisfy `sqft_living = sqft_above + sqft_basement`, which holds in 100% of the data.
- **`weights="distance"` relies heavily on exact matches** (distance 0), which are common in this discrete data. Results were verified to be deterministic for this dataset and the pinned scikit-learn version.
- **No parallelism:** one uvicorn process, and an `async def` handler doing blocking CPU work.
- **Container and security gaps:** the container runs as root and has no Docker `HEALTHCHECK` (docker-compose defines one). CORS allows `*` together with credentials. There is no authentication or rate limiting.
- `make test-integration` binds host port 8000.
- A `--no-cache` image build failed twice during development because a Debian mirror returned a file with the wrong hash in the provided `apt-get install curl` step. Cached builds were unaffected.

This is a solid local deliverable with tests and measurements. It is **not** a fully production-ready service; see the next steps.

## Deliberately out of scope

- Retraining or changing the model.
- Choosing a different imputation method.
- Adding scaling to KNN.
- Changing the server's concurrency model.
- Returning imputed fields in the response.
- Checking that the area fields are consistent with each other.
- Deleting the duplicate `create_model.py`.
- Any external infrastructure (cache, database, queue, cloud deployment, metrics stack, tracing).

None of these are needed to meet the assignment's requirements, and each would add risk or scope without measured need.

## Possible production next steps

1. **Concurrency:** a sync handler (thread pool) and/or several uvicorn workers. Measure with `benchmarks/benchmark.py`, since the model's `predict` uses CPU.
2. **Extend the missing-data evaluation** to realistic missingness patterns and other rates. Review scaling before KNN with the Data Science team, starting from the `sqft_lot` result.
3. **Version the model, the imputer reference data and the dependencies together**, e.g. a saved imputer artifact plus a model registry.
4. **Harden the container:** non-root user, Docker `HEALTHCHECK`, explicit CORS origins, pinned test dependencies.
5. **Observability:** structured logs and metrics for latency, imputation rate per field, and unknown zipcodes.
6. **CI** running `make test-all` on every change.

---

## Project structure

```
├── Dockerfile / Dockerfile.test / docker-compose.test.yml / Makefile
├── requirements.txt            # pinned runtime dependencies
├── requirements-test.txt       # pinned test dependencies
├── benchmarks/benchmark.py     # latency/throughput benchmark
├── create_model.py             # training script (used by src/model/Dockerfile)
├── notebooks
│   ├── imputation_experiment.ipynb   # Data Science team's research (provided)
│   ├── final_evaluation.ipynb        # held-out evaluation + benchmark plots
│   └── requirements-notebook.txt     # extra deps to run the evaluation notebook
├── src
│   ├── main.py                 # FastAPI app, logging, 422 handler
│   ├── api/endpoints.py        # schema, resource loading, /health, /predict
│   ├── utils/loader.py         # load model/features/demographics, fit imputer
│   ├── model/                  # model.pkl, model_features.json (unchanged)
│   └── data/                   # kc_house_data.csv, zipcode_demographics.csv, future_unseen_examples.csv
└── test
    ├── conftest.py
    ├── fixtures/baseline_predictions.json   # original API predictions (parity reference)
    ├── unit/                   # imputation, regression, robustness, performance
    └── integration/            # tests against the running container
```

---

## AI-assisted development

This work was done with an AI coding assistant (Claude Code) and guided by a human. The work was split into stages:

1. Measure the baseline.
2. Design the imputation.
3. Minimal imputation implementation.
4. Performance.
5. Production-readiness pass.
6. Tests.
7. Final benchmark.
8. Documentation.

The assistant worked under explicit rules: keep the model, treat KNN as a given requirement, keep changes small, never invent numbers, and stop for a decision whenever ML behavior was ambiguous.

**What the AI was used for:**
- Reading the codebase and notebook.
- Tracing the request path.
- Writing measurement scripts.
- Implementing each stage.
- Writing tests and running them in Docker.
- Drafting this documentation.

**How suggestions were reviewed:**
- Every stage ended with a report: files changed, tests run and their results, assumptions, risks, and open decisions. The next stage started only after that report.
- The imputation design (feature set, reference data, all-missing behavior, error codes, response contract) was presented as explicit options with a recommendation before any code was written.

**How changes were validated:**
- **Profiling before optimizing:** each step of the original handler was timed first.
- **Exact parity:** predictions were compared byte for byte against the original Docker image, both for complete requests and for 300 deterministic incomplete requests across stages.
- **Interleaved benchmarks:** both images were run back to back in alternating rounds.
- **Deliberately broken code** confirmed that the new tests detect real regressions.

**Where the evidence changed or simplified the plan:**
- **Expected bottleneck vs measured one:** the brief's hint pointed to per-request file loading. Measurement showed loading was ~1 ms, while a debug `print()` was ~6 ms of an ~11 ms request. Both were fixed, and the docs report the actual breakdown.
- **"Equivalent" imputer configurations weren't equivalent:** in theory, fitting the imputer on all 17 notebook columns should match fitting on the 7 API columns. In practice, results differed in 24 of 100 test rows. In the example investigated, numerical rounding turned an exact (zero-distance) match into a distance of 0.000161, which changed the neighbor weights. That led to the 7-column design and the version pinning.
- **Scaling for KNN was proposed and rejected,** because it would change the Data Science team's method without evaluation. It's documented as a trade-off.
- **Infrastructure kept out of scope:** a FastAPI lifespan refactor, a service layer, multiple workers, response-schema changes and external infrastructure were all considered and left out (see above).

**Mistakes the tests caught:**
- **`NaN` regression:** one refactoring turned `NaN` inputs from "treated as missing" into a 500 error. It was found while adding validation, then fixed and covered by tests.
- **Wrong attribute name:** a startup log line used an attribute KNNImputer doesn't have, so every test failed at import until it was fixed.
- **Validation crash:** a "one-line" `NaN` rejection turned out to crash FastAPI's default error handler. Tests exposed it, which led to the small custom 422 handler.
