# Monitoring the RAG Gateway

The Gateway exposes Prometheus metrics, optional OpenTelemetry traces, and a
durable SQLite model-usage ledger. Grafana can visualize metrics from either
Prometheus or InfluxDB, and traces from a tracing backend such as Grafana Tempo.
Prometheus and InfluxDB do not store the traces themselves.

## 1. Gateway configuration

Install the project's dependencies again after updating:

```powershell
python -m pip install -e ".[dev,loader]"
```

Set these values in `.env`, or use the authenticated environment settings page:

```dotenv
OPERATIONS_DB_PATH=data/rag_gateway_operations.sqlite3
OTEL_TRACES_ENABLED=false
OTEL_SERVICE_NAME=rag-gateway
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
OTEL_TRACES_SAMPLE_RATIO=1.0
MODEL_PRICING={}
```

Restart the Gateway after changing settings. Usage is persisted even when
tracing is disabled or a trace is not sampled. Set `OTEL_TRACES_ENABLED=true`
only after configuring an approved collector. The endpoint must be the full
OTLP **HTTP/protobuf** traces URL, not a gRPC endpoint.

`GATEWAY_API_KEY` is required to read `/metrics` and `/v1/admin/usage`. Registered
tenant keys cannot read these administrator-wide endpoints. This intentionally
tightens the previous `/metrics` authorization because its new series contain
cross-tenant usage and cost data. Never put the administrator key in dashboards,
browser URLs, committed configuration, or metric labels.

Validate using an environment variable that contains your administrator key:

```powershell
$headers = @{ Authorization = "Bearer $env:GATEWAY_API_KEY" }
Invoke-WebRequest http://127.0.0.1:8000/metrics -Headers $headers
Invoke-RestMethod http://127.0.0.1:8000/v1/admin/usage -Headers $headers
Invoke-RestMethod "http://127.0.0.1:8000/v1/admin/usage?tenant_id=default" -Headers $headers
```

The usage endpoint returns `currency: "USD"` and lifetime aggregates grouped by
`tenant_id`, `provider`, `operation`, and the actual upstream `model`, not the
Gateway's public model alias. It reports calls, failed/incomplete calls, known
token counts, known estimated costs, and counts of missing usage/unpriced calls.

### Pricing and coverage

`MODEL_PRICING` is a JSON object keyed by **provider:operation:model**. The
following rates are **illustrative, not vendor prices**:

```dotenv
MODEL_PRICING={"mongodb_ai:embedding:voyage-4":{"input_per_million_usd":0.1},"ollama:generation:qwen3.5:9b":{"input_per_million_usd":1,"output_per_million_usd":2},"atlas:rerank:rerank-2.5":{"per_request_usd":0.001}}
```

Use your contracted prices, or explicit zero input/output rates for a model you
intend to treat as free. Unconfigured prices do not imply zero cost.

- Token-based cost is `input_tokens * input_per_million_usd / 1e6 +
  output_tokens * output_per_million_usd / 1e6`.
- `per_request_usd` adds a fixed charge per recorded invocation. With no
  token-based rates, it is the entire fixed-price estimate. If token-based rates
  are also configured, their required counts must be available.
- Without a fixed price, missing rates for nonzero token components or missing
  input/output counts make the cost unknown. Negative and non-finite prices and
  unknown pricing fields are rejected.
- `mongodb_ai:embedding:<model>` records each embedding HTTP attempt, including
  retries, in ingestion and query paths. Reported `usage.total_tokens` is the
  embedding input count; no character-to-token guess is persisted.
- `ollama:generation:<model>` records one call per streamed or non-streamed
  generation. Streams request `stream_options.include_usage`; the last upstream
  usage snapshot is saved once, not added repeatedly. An older upstream may
  omit usage; the ledger then records missing usage. Failed calls and streams
  without `[DONE]` are retained as failed/incomplete rather than zero usage.
- `ollama:evaluation_judge:<model>` records each local answer-quality judge call;
  its answer generation is recorded separately as `ollama:generation:<model>`.
- `atlas:rerank:<model>` records native reranking aggregations. Atlas does not
  return reranking token usage through this path, so token counts stay unknown.
  Configure a fixed per-invocation estimate only if appropriate to your pricing
  agreement; it is not an authoritative charge or a document-count tariff.
- `ollama:ocr:<model>` records native Ollama `prompt_eval_count` and `eval_count`
  for document-image/PDF OCR. Loader jobs inherit the job tenant. Preview OCR,
  which has no tenant selection, is stored with an empty `tenant_id`
  (unattributed).
- Startup smoke checks, model administration, and standalone CLI ingestion
  outside the running Gateway telemetry context are not included.

The `model_usage` table stores timestamps, request/trace IDs, token counts,
outcome, the pricing snapshot, and the estimated cost at call time. Later price
changes do not reprice historical events. Protect and back up the SQLite file;
there is no automatic usage retention/deletion. Persistence failures log an
explicit error and propagate; these records are operational estimates, not an
invoice or guaranteed exactly-once billing through crashes.

### Available metrics

| Metric family | Labels / meaning |
| --- | --- |
| `rag_gateway_http_requests_total` | `method`, route template, actual HTTP `status` |
| `rag_gateway_http_request_duration_seconds_sum`, `_count` | HTTP mean latency inputs; process-local |
| `rag_gateway_processing_stage_duration_seconds_bucket`, `_sum`, `_count` | Stage histograms labeled by `stage` |
| `rag_gateway_model_calls_total` | Durable recorded invocations |
| `rag_gateway_model_failed_calls_total` | Durable failed/incomplete invocations |
| `rag_gateway_model_tokens_total` | Known token sums; extra `token_type=input\|output\|total` |
| `rag_gateway_model_estimated_cost_usd_total` | Sum of known estimates, not a complete bill |
| `rag_gateway_model_missing_usage_calls_total` | Calls without a reported/derivable total token count |
| `rag_gateway_model_unpriced_calls_total` | Calls without a complete cost estimate |

Model metrics have `tenant_id`, `provider`, `operation`, and `model` labels.
Unknown token/cost values are omitted, not exported as fabricated zeros. Always
show missing-usage and unpriced-call panels next to token and cost panels.
SQLite aggregates survive Gateway restarts; HTTP and stage counters do not.
Prometheus rate/increase queries need multiple scrapes and do not reconstruct
usage before scraping began. The usage API is the lifetime source of truth.

Use one Gateway worker for reliable process-local request metrics. Scraping a
load balancer that switches between workers produces misleading rates. Do not
sum model metrics from several instances sharing the same operations DB: each
exports the same ledger totals. Independent Gateway instances with independent
databases can be summed. Bound tenant/model cardinality and restrict who can
create new tenants or configure model names.

## 2. Option A: Prometheus and Grafana

Run Prometheus and Grafana on your approved monitoring infrastructure. Save the
administrator token in a permissions-restricted file on the **Prometheus host**,
for example `C:\monitoring\secrets\gateway-token.txt` on Windows. The file should
contain only the token. Substitute paths and targets in this Prometheus config:

```yaml
global:
  scrape_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: rag-gateway
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: 'C:\monitoring\secrets\gateway-token.txt'
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

For Linux/container monitoring, mount the token read-only and use its
container-local path, such as `/run/secrets/gateway-token`. `127.0.0.1` refers to
the scraper itself: use the Gateway service DNS or an approved host address
when services are in different containers/hosts. Use HTTPS/TLS verification
when traffic leaves a trusted local network. Do not disable certificate checks.

1. Validate with `promtool check config <your-config>` and start/reload Prometheus.
2. Open Prometheus **Status > Targets** and verify `rag-gateway` is UP.
3. In Grafana, add a **Prometheus** data source, using the Prometheus URL reachable
   from the Grafana server, then **Save & test**.
4. Create a dashboard; add panels using the queries below. Set the Prometheus
   data source and an appropriate unit on every panel.
5. Add `$tenant` and `$model` dashboard variables from label values, and use
   `tenant_id=~"$tenant",model=~"$model"` filters on model metrics as needed.

### Suggested PromQL panels

**Requests/s**, time series:

```promql
sum by (route) (rate(rag_gateway_http_requests_total{job="rag-gateway"}[$__rate_interval]))
```

**HTTP 5xx fraction**, percent (0-1):

```promql
sum(rate(rag_gateway_http_requests_total{job="rag-gateway",status=~"5.."}[$__rate_interval]))
/
clamp_min(sum(rate(rag_gateway_http_requests_total{job="rag-gateway"}[$__rate_interval])), 0.000001)
```

This is the transport error rate. A stream can fail after HTTP 200; use the model
failure panel or the live monitoring page to see those application failures.

**Mean HTTP latency**, seconds:

```promql
sum by (route) (rate(rag_gateway_http_request_duration_seconds_sum{job="rag-gateway"}[$__rate_interval]))
/
sum by (route) (rate(rag_gateway_http_request_duration_seconds_count{job="rag-gateway"}[$__rate_interval]))
```

**p95 stage latency**, seconds:

```promql
histogram_quantile(0.95,
  sum by (le, stage) (rate(rag_gateway_processing_stage_duration_seconds_bucket{job="rag-gateway"}[$__rate_interval]))
)
```

`atlas_vector_rerank` is a combined vector/rerank measurement for vector-only
retrieval. Hybrid reranking runs after fusion and is measured separately as
`atlas_hybrid_rerank`. These are client-visible aggregation durations, not
database-internal timings. `retrieval` includes its embedding/Atlas child
stages: do not stack the parent and children together. Buckets have finite
resolution; no requests means no percentile, not zero latency.

**Tokens over selected interval**, stat/table:

```promql
sum by (tenant_id, model, operation) (
  increase(rag_gateway_model_tokens_total{job="rag-gateway",token_type="total"}[$__range])
)
```

**Known estimated cost over selected interval**, currency USD:

```promql
sum by (tenant_id, model, operation) (
  increase(rag_gateway_model_estimated_cost_usd_total{job="rag-gateway"}[$__range])
)
```

**Unpriced calls**, **missing usage**, and **failed/incomplete model calls**:
use the following expressions as separate stat/table panels.

```promql
sum by (tenant_id, model, operation) (increase(rag_gateway_model_unpriced_calls_total{job="rag-gateway"}[$__range]))
sum by (tenant_id, model, operation) (increase(rag_gateway_model_missing_usage_calls_total{job="rag-gateway"}[$__range]))
sum by (tenant_id, model, operation) (increase(rag_gateway_model_failed_calls_total{job="rag-gateway"}[$__range]))
```

For lifetime totals, use an **instant** query without `increase`, such as
`sum by (tenant_id, model) (rag_gateway_model_calls_total{job="rag-gateway"})`.
Never sum all three token types: that would double-count the input/output.

## 3. Option B: InfluxDB, Telegraf and Grafana

This concrete configuration targets **InfluxDB OSS 2.x with Flux**. InfluxDB 3.x
does not support Flux; use its SQL/InfluxQL Grafana mode instead and adapt the
queries. The Gateway does not push directly into InfluxDB: Telegraf scrapes the
same authenticated `/metrics` endpoint.

Create an InfluxDB organization and bucket (example: `monitoring` / `rag-gateway`)
with a suitable retention policy. Give Telegraf a write-only token and Grafana a
separate read-only token. Save the Gateway administrator key in a restricted
file on the Telegraf host. Example Telegraf configuration:

```toml
[agent]
  interval = "15s"
  flush_interval = "15s"

[[inputs.prometheus]]
  urls = ["http://127.0.0.1:8000/metrics"]
  metric_version = 2
  name_override = "rag_gateway"
  bearer_token = 'C:\monitoring\secrets\gateway-token.txt'
  response_timeout = "10s"

[[outputs.influxdb_v2]]
  urls = ["http://127.0.0.1:8086"]
  token = "${INFLUX_TOKEN}"
  organization = "monitoring"
  bucket = "rag-gateway"
```

Supply `INFLUX_TOKEN` via your process environment/secret manager. Adjust URLs
and token paths for containers as described in the Prometheus section.
Run `telegraf --config <your-config> --test` to inspect parsed metrics (this does
not write to InfluxDB), then run Telegraf normally.

With `metric_version=2`, the measurement is `rag_gateway` and Prometheus sample
names become fields. Labels become tags, including `tenant_id`, `model`,
`operation`, `provider`, `token_type`, and histogram `le`. Verify the actual
field names in your installed Telegraf before saving panels.

In Grafana add an **InfluxDB** data source, select **Flux**, set the InfluxDB URL,
organization, read token and default bucket, then **Save & test**.

### Flux examples

**Lifetime known cost by tenant/model**, stat/table (do not sum scrape snapshots):

```flux
from(bucket: "rag-gateway")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "rag_gateway" and
    r._field == "rag_gateway_model_estimated_cost_usd_total")
  |> last()
  |> group(columns: ["tenant_id", "model"])
  |> sum(column: "_value")
```

**Token rate**, time series; take derivatives per original series before any
cross-tenant/model aggregation:

```flux
from(bucket: "rag-gateway")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "rag_gateway" and
    r._field == "rag_gateway_model_tokens_total" and r.token_type == "total")
  |> aggregateWindow(every: v.windowPeriod, fn: last, createEmpty: false)
  |> derivative(unit: 1s, nonNegative: true)
```

For missing usage, failures, and unpriced calls, substitute their metric names.
For interval cost, use a derivative/increase treatment rather than summing
cumulative samples. Sparse/new series and counter resets affect windowed
estimates; compare lifetime totals with `/v1/admin/usage`.

For an initial stage dashboard, graph the `_sum` and `_count` fields and divide
their non-negative rates for average seconds by stage. Keep the `stage` tags.
For percentile panels, preserve histogram `le` tags and use a tested histogram
query for your InfluxDB version, or prefer Prometheus's `histogram_quantile`.

## 4. OpenTelemetry traces and Grafana Tempo

An approved OpenTelemetry Collector can receive the Gateway's traces and forward
them to a self-hosted Tempo deployment. Example collector pipeline:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318

processors:
  batch: {}

exporters:
  otlphttp/tempo:
    endpoint: http://tempo:4318

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/tempo]
```

Here `tempo` is a placeholder for your Tempo host/service, with its OTLP HTTP
receiver enabled. Bind the collector to an appropriate private interface if the
Gateway is remote/in another container; configure TLS/authentication at that
boundary. Grafana's Tempo data source connects to Tempo's query endpoint
(commonly port 3200), **not** to the collector's 4318 ingestion port.

Enable the Gateway configuration from section 1. The sampling ratio is for new
root traces; valid upstream W3C `traceparent`/`tracestate` sampling decisions are
respected. The Gateway forwards this context to embedding and Ollama HTTP calls.
It does not forward baggage or copy inbound headers into span attributes.
The batch exporter flushes/shuts down on normal Gateway shutdown. Collector
outages produce exporter errors but do not disable local usage accounting.

In Grafana Explore, choose Tempo and search for service `rag-gateway`, or paste
the trace ID from the monitoring request details/JSON request log. The span
attribute `rag.request_id` correlates with the response's `X-Request-ID`.
Request spans use route templates. Child spans cover retrieval, embedding
batches, Atlas vector/lexical search and generation, including streaming.
Native reranking is represented by `atlas_vector_rerank` for vector-only retrieval
or `atlas_hybrid_rerank` for the separate post-fusion hybrid aggregation, with
model attributes. No artificial split of a combined aggregation is claimed.

Traces do not include prompts, answers, source text, raw URLs, credentials or
exception messages/stack traces. Error spans retain safe exception type/status.
Only send telemetry to infrastructure approved for request IDs, model names,
tenant usage and operational metadata. Avoid additional automatic HTTP/database
instrumentation that might capture bodies, full URLs, or MongoDB query vectors.

## 5. Troubleshooting and operational limits

- **401 / target DOWN:** use the administrator key, not a registered tenant key;
  verify the secret file is readable by the scraper process.
- **Connection refused:** verify host/container networking and the Gateway bind
  address. A service bound only to localhost is not reachable remotely.
- **No model series yet:** perform an ingestion or chat request. Series appear
  only after the first recorded call; absent data is not zero usage.
- **Unknown cost:** check exact provider/operation/model keys, required rates,
  upstream token availability and `unpriced_calls`.
- **Missing stream usage:** inspect the upstream version and support for
  `stream_options.include_usage`; no token count is guessed from text.
- **No trace:** check `OTEL_TRACES_ENABLED`, collector reachability, `/v1/traces`
  endpoint, sampling, and exporter errors. A parent can request no sampling.
- **DB errors:** verify writable persistent storage, free space and filesystem
  permissions. Do not share SQLite over an unsupported network filesystem.
- **Unexpected totals:** do not duplicate scraping of one shared ledger, add
  cumulative scrape samples, or sum token total with its input/output components.
- Add Grafana alert rules for target-down, error rate and missing/unpriced
  usage after establishing a baseline. No preconfigured alert rules are shipped.

## References

- [Prometheus scrape configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Telegraf Prometheus input and metric mapping](https://docs.influxdata.com/telegraf/v1/input-plugins/prometheus/)
- [Grafana InfluxDB data source configuration](https://grafana.com/docs/grafana/latest/datasources/influxdb/configure/)
- [OpenTelemetry Collector configuration](https://opentelemetry.io/docs/collector/configuration/)
- [Grafana Tempo configuration](https://grafana.com/docs/tempo/latest/configuration/)
