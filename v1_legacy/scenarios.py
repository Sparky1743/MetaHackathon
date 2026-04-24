"""
Scenario definitions for OnCallEnv.

Each scenario defines:
  - Infrastructure topology (services, dependencies, configs)
  - The injected fault / incident
  - Expected investigation path
  - Root cause and correct remediation
  - Grading rubric with partial-credit keys
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ServiceDef:
    name: str
    healthy: bool = True
    cpu: float = 25.0
    memory: float = 40.0
    latency_p50: float = 12.0
    latency_p99: float = 45.0
    error_rate: float = 0.1
    rps: float = 500.0
    extra_metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    deploy_history: list[dict] = field(default_factory=list)


@dataclass
class Scenario:
    task_id: str
    task_name: str
    difficulty: str
    goal: str
    description: str
    max_steps: int
    services: dict[str, ServiceDef] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    root_cause_service: str = ""
    root_cause_keywords: list[str] = field(default_factory=list)
    correct_remediation: str = ""           # action command prefix
    correct_remediation_args: dict = field(default_factory=dict)
    investigation_targets: list[str] = field(default_factory=list)  # services worth checking
    penalty_services: list[str] = field(default_factory=list)      # wrong services to act on
    # Multi-root-cause scenarios: list of {cmd, service, key} dicts
    correct_remediations: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  EASY — Memory Leak in Payment Service
# ═══════════════════════════════════════════════════════════════════════════════

def build_easy() -> Scenario:
    return Scenario(
        task_id="easy_memory_leak",
        task_name="Memory Leak in Payment Service",
        difficulty="easy",
        goal=(
            "ALERT: payment-service is experiencing OOM (Out-Of-Memory) kills. "
            "Investigate the issue, identify the root cause, and take corrective action "
            "to restore the service."
        ),
        description=(
            "A single service (payment-service) has a memory leak causing repeated "
            "OOM kills. The agent needs to check logs, confirm the diagnosis, and "
            "restart the service."
        ),
        max_steps=10,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                cpu=30.0, memory=45.0, latency_p50=15.0, latency_p99=80.0,
                error_rate=2.5, rps=1200.0,
                logs=[
                    "2025-04-01T10:01:12Z INFO  api-gateway  Request routed to payment-service",
                    "2025-04-01T10:01:14Z WARN  api-gateway  Upstream payment-service returned 503",
                    "2025-04-01T10:01:15Z WARN  api-gateway  Upstream payment-service returned 503",
                    "2025-04-01T10:02:00Z INFO  api-gateway  Retry succeeded for payment-service",
                    "2025-04-01T10:03:22Z WARN  api-gateway  Upstream payment-service returned 503",
                ],
                config={"timeout_ms": 5000, "retry_count": 3, "port": 8080},
                dependencies=["payment-service", "user-service", "inventory-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "payment-service": ServiceDef(
                name="payment-service",
                healthy=False,
                cpu=45.0, memory=94.7, latency_p50=850.0, latency_p99=4200.0,
                error_rate=18.3, rps=180.0,
                extra_metrics={"oom_kills_last_hour": 7, "restart_count_24h": 12,
                               "heap_used_mb": 1820, "heap_max_mb": 2048},
                logs=[
                    "2025-04-01T09:45:00Z INFO  payment-service  Service started (v3.2.1)",
                    "2025-04-01T09:52:33Z WARN  payment-service  GC overhead limit exceeded — heap at 1.6GB",
                    "2025-04-01T09:55:10Z WARN  payment-service  Memory pressure: allocating from swap",
                    "2025-04-01T09:58:44Z ERROR payment-service  java.lang.OutOfMemoryError: Java heap space",
                    "2025-04-01T09:58:44Z ERROR payment-service  at com.pay.cache.TransactionCache.put(TransactionCache.java:142)",
                    "2025-04-01T09:58:45Z ERROR payment-service  Container killed by OOM killer (exit code 137)",
                    "2025-04-01T09:58:50Z INFO  payment-service  Service restarting...",
                    "2025-04-01T10:00:01Z INFO  payment-service  Service started (v3.2.1)",
                    "2025-04-01T10:03:15Z WARN  payment-service  GC overhead limit exceeded — heap at 1.7GB",
                    "2025-04-01T10:05:30Z ERROR payment-service  java.lang.OutOfMemoryError: Java heap space",
                    "2025-04-01T10:05:31Z ERROR payment-service  Container killed by OOM killer (exit code 137)",
                ],
                config={
                    "jvm_heap_max": "2048m",
                    "transaction_cache_ttl_sec": 86400,
                    "transaction_cache_max_entries": "unlimited",
                    "db_pool_size": 20,
                    "port": 8081,
                },
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v3.1.0", "date": "2025-03-15", "status": "stable"},
                    {"version": "v3.2.0", "date": "2025-03-28", "status": "stable",
                     "notes": "Added transaction caching layer"},
                    {"version": "v3.2.1", "date": "2025-03-31", "status": "current",
                     "notes": "Bug fix for cache serialization"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=20.0, memory=35.0, latency_p50=8.0, latency_p99=30.0,
                error_rate=0.05, rps=800.0,
                logs=[
                    "2025-04-01T10:00:00Z INFO  user-service  Health check OK",
                    "2025-04-01T10:01:00Z INFO  user-service  Health check OK",
                ],
                config={"db_pool_size": 15, "port": 8082},
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                cpu=35.0, memory=60.0, latency_p50=2.0, latency_p99=15.0,
                error_rate=0.0, rps=2000.0,
                extra_metrics={"connections_active": 38, "connections_max": 100,
                               "replication_lag_ms": 12},
                logs=[
                    "2025-04-01T10:00:00Z INFO  postgres  Checkpoint complete",
                ],
                config={"max_connections": 100, "shared_buffers": "4GB"},
                dependencies=[],
                deploy_history=[],
            ),
        },
        alerts=[
            {"alert_id": "ALT-001", "severity": "critical",
             "service": "payment-service",
             "message": "OOM killer invoked 7 times in the last hour",
             "timestamp": "2025-04-01T10:05:31Z"},
            {"alert_id": "ALT-002", "severity": "warning",
             "service": "payment-service",
             "message": "Memory usage at 94.7% — above 90% threshold",
             "timestamp": "2025-04-01T10:03:15Z"},
            {"alert_id": "ALT-003", "severity": "warning",
             "service": "api-gateway",
             "message": "Elevated 503 error rate from upstream payment-service",
             "timestamp": "2025-04-01T10:01:14Z"},
        ],
        root_cause_service="payment-service",
        root_cause_keywords=["memory", "oom", "leak", "heap", "cache", "outofmemory"],
        correct_remediation="restart_service",
        correct_remediation_args={"service": "payment-service"},
        investigation_targets=["payment-service", "api-gateway"],
        penalty_services=["user-service", "postgres-primary"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIUM — Cascading Database Connection Pool Exhaustion
# ═══════════════════════════════════════════════════════════════════════════════

def build_medium() -> Scenario:
    return Scenario(
        task_id="medium_cascading_failure",
        task_name="Cascading Connection Pool Exhaustion",
        difficulty="medium",
        goal=(
            "ALERT: api-gateway is reporting high latency and timeouts. Multiple "
            "services appear affected. Trace the dependency chain, identify the "
            "root cause, and apply the correct fix."
        ),
        description=(
            "The API gateway is timing out. Root cause: a recent config change "
            "reduced the DB connection pool size in order-service, causing connection "
            "starvation that cascades upstream."
        ),
        max_steps=15,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                healthy=False,
                cpu=55.0, memory=50.0, latency_p50=2200.0, latency_p99=8500.0,
                error_rate=35.0, rps=400.0,
                logs=[
                    "2025-04-01T14:00:10Z WARN  api-gateway  Upstream order-service timeout after 5000ms",
                    "2025-04-01T14:00:12Z WARN  api-gateway  Upstream order-service timeout after 5000ms",
                    "2025-04-01T14:00:15Z ERROR api-gateway  Circuit breaker OPEN for order-service",
                    "2025-04-01T14:00:30Z WARN  api-gateway  Upstream notification-service slow (3200ms)",
                    "2025-04-01T14:01:00Z ERROR api-gateway  5xx rate at 35% — SLA breach imminent",
                    "2025-04-01T14:01:30Z WARN  api-gateway  Thread pool nearing capacity (180/200)",
                ],
                config={"timeout_ms": 5000, "circuit_breaker_threshold": 5, "thread_pool": 200, "port": 8080},
                dependencies=["order-service", "user-service", "notification-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "order-service": ServiceDef(
                name="order-service",
                healthy=False,
                cpu=80.0, memory=70.0, latency_p50=4500.0, latency_p99=9800.0,
                error_rate=42.0, rps=50.0,
                extra_metrics={"db_pool_active": 5, "db_pool_max": 5,
                               "db_pool_pending_requests": 312,
                               "thread_pool_active": 195, "thread_pool_max": 200},
                logs=[
                    "2025-04-01T13:45:00Z INFO  order-service  Config reload triggered",
                    "2025-04-01T13:45:01Z INFO  order-service  db_pool_size changed: 50 -> 5",
                    "2025-04-01T13:50:22Z WARN  order-service  Connection pool exhausted — 50 requests queued",
                    "2025-04-01T13:55:10Z ERROR order-service  Timeout waiting for DB connection (30s)",
                    "2025-04-01T14:00:00Z ERROR order-service  Connection pool exhausted — 312 requests queued",
                    "2025-04-01T14:00:05Z ERROR order-service  org.postgresql.util.PSQLException: Cannot get connection",
                    "2025-04-01T14:00:08Z ERROR order-service  Request processing failed: DB connection timeout",
                    "2025-04-01T14:00:30Z WARN  order-service  Thread starvation detected",
                ],
                config={
                    "db_pool_size": 5,
                    "db_pool_max_wait_ms": 30000,
                    "port": 8083,
                    "thread_pool_size": 200,
                    "config_last_modified": "2025-04-01T13:45:00Z",
                    "config_modified_by": "deploy-bot (auto-scaler rule #47)",
                },
                dependencies=["postgres-primary", "redis-cache"],
                deploy_history=[
                    {"version": "v4.8.0", "date": "2025-03-25", "status": "stable"},
                    {"version": "v4.8.0", "date": "2025-04-01", "status": "current",
                     "notes": "Config change: auto-scaler adjusted db_pool_size"},
                ],
            ),
            "notification-service": ServiceDef(
                name="notification-service",
                healthy=False,
                cpu=60.0, memory=55.0, latency_p50=1800.0, latency_p99=5500.0,
                error_rate=15.0, rps=100.0,
                logs=[
                    "2025-04-01T14:00:05Z WARN  notification-service  Dependency order-service responding slowly",
                    "2025-04-01T14:00:10Z WARN  notification-service  Timeout calling order-service for order details",
                    "2025-04-01T14:00:30Z ERROR notification-service  Failed to send order confirmation: upstream timeout",
                ],
                config={"order_service_timeout_ms": 3000, "port": 8084},
                dependencies=["order-service"],
                deploy_history=[
                    {"version": "v2.1.0", "date": "2025-03-18", "status": "stable"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=22.0, memory=38.0, latency_p50=10.0, latency_p99=35.0,
                error_rate=0.1, rps=600.0,
                logs=[
                    "2025-04-01T14:00:00Z INFO  user-service  Health check OK",
                ],
                config={"db_pool_size": 30, "port": 8082},
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                cpu=40.0, memory=65.0, latency_p50=3.0, latency_p99=18.0,
                error_rate=0.0, rps=1500.0,
                extra_metrics={"connections_active": 42, "connections_max": 100,
                               "replication_lag_ms": 8},
                logs=[
                    "2025-04-01T14:00:00Z INFO  postgres  Connection count normal (42/100)",
                ],
                config={"max_connections": 100, "shared_buffers": "4GB"},
                dependencies=[],
                deploy_history=[],
            ),
            "redis-cache": ServiceDef(
                name="redis-cache",
                cpu=15.0, memory=50.0, latency_p50=1.0, latency_p99=5.0,
                error_rate=0.0, rps=5000.0,
                extra_metrics={"hit_rate_percent": 92.0, "evictions_per_sec": 2},
                logs=[
                    "2025-04-01T14:00:00Z INFO  redis  Memory usage normal",
                ],
                config={"maxmemory": "2gb", "maxmemory-policy": "allkeys-lru"},
                dependencies=[],
                deploy_history=[],
            ),
        },
        alerts=[
            {"alert_id": "ALT-101", "severity": "critical",
             "service": "api-gateway",
             "message": "Error rate at 35% — SLA breach",
             "timestamp": "2025-04-01T14:01:00Z"},
            {"alert_id": "ALT-102", "severity": "critical",
             "service": "order-service",
             "message": "Database connection pool exhausted — 312 requests queued",
             "timestamp": "2025-04-01T14:00:00Z"},
            {"alert_id": "ALT-103", "severity": "warning",
             "service": "notification-service",
             "message": "Elevated latency — p99 at 5500ms",
             "timestamp": "2025-04-01T14:00:30Z"},
            {"alert_id": "ALT-104", "severity": "warning",
             "service": "api-gateway",
             "message": "Circuit breaker OPEN for order-service",
             "timestamp": "2025-04-01T14:00:15Z"},
        ],
        root_cause_service="order-service",
        root_cause_keywords=["pool", "connection", "db_pool_size", "config", "5", "exhausted", "auto-scaler"],
        correct_remediation="update_config",
        correct_remediation_args={"service": "order-service", "key": "db_pool_size"},
        investigation_targets=["api-gateway", "order-service", "notification-service"],
        penalty_services=["user-service", "redis-cache"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HARD — Subtle Cache Deployment Bug Causing Cross-Service Degradation
# ═══════════════════════════════════════════════════════════════════════════════

def build_hard() -> Scenario:
    return Scenario(
        task_id="hard_cache_degradation",
        task_name="Subtle Cache Bug Causing Cross-Service Degradation",
        difficulty="hard",
        goal=(
            "ALERT: Multiple services are reporting slightly elevated latency and "
            "increased database load. No single service is critically failing. "
            "Find the root cause of the system-wide performance degradation and fix it."
        ),
        description=(
            "A recent deployment to cache-service introduced a key-hashing bug that "
            "causes ~60% cache miss rate (normally 5%). This silently pushes all "
            "services to hit the database directly, increasing DB load and latency "
            "across the board. The tricky part: no single service is 'down' and "
            "the cache-service itself reports healthy."
        ),
        max_steps=20,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                cpu=45.0, memory=50.0, latency_p50=120.0, latency_p99=850.0,
                error_rate=3.2, rps=900.0,
                logs=[
                    "2025-04-01T18:00:00Z INFO  api-gateway  Routing normal",
                    "2025-04-01T18:05:00Z WARN  api-gateway  Aggregate p99 latency elevated: 850ms (baseline: 200ms)",
                    "2025-04-01T18:10:00Z WARN  api-gateway  Slow responses from order-service and product-service",
                ],
                config={"timeout_ms": 5000, "port": 8080},
                dependencies=["order-service", "product-service", "user-service", "search-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "order-service": ServiceDef(
                name="order-service",
                cpu=55.0, memory=60.0, latency_p50=95.0, latency_p99=680.0,
                error_rate=1.8, rps=400.0,
                extra_metrics={"db_queries_per_request": 8.2, "db_queries_baseline": 2.1,
                               "cache_calls_per_request": 6, "cache_miss_rate_percent": 62.0},
                logs=[
                    "2025-04-01T18:00:10Z INFO  order-service  Processing orders normally",
                    "2025-04-01T18:02:00Z WARN  order-service  Cache miss rate elevated: 62% (baseline: 5%)",
                    "2025-04-01T18:03:00Z WARN  order-service  DB query count per request: 8.2 (baseline: 2.1)",
                    "2025-04-01T18:05:00Z INFO  order-service  Increased DB pool utilization: 85%",
                ],
                config={"db_pool_size": 30, "cache_ttl_sec": 300, "port": 8083},
                dependencies=["postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v4.8.0", "date": "2025-03-25", "status": "stable"},
                ],
            ),
            "product-service": ServiceDef(
                name="product-service",
                cpu=50.0, memory=55.0, latency_p50=85.0, latency_p99=620.0,
                error_rate=1.5, rps=600.0,
                extra_metrics={"db_queries_per_request": 5.5, "db_queries_baseline": 1.4,
                               "cache_calls_per_request": 4, "cache_miss_rate_percent": 58.0},
                logs=[
                    "2025-04-01T18:01:00Z WARN  product-service  Cache miss rate elevated: 58%",
                    "2025-04-01T18:03:00Z WARN  product-service  Slow DB responses — avg 45ms (baseline 8ms)",
                    "2025-04-01T18:05:00Z INFO  product-service  No errors in application logic",
                ],
                config={"db_pool_size": 25, "cache_ttl_sec": 600, "port": 8085},
                dependencies=["postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v3.2.0", "date": "2025-03-22", "status": "stable"},
                ],
            ),
            "search-service": ServiceDef(
                name="search-service",
                cpu=48.0, memory=52.0, latency_p50=110.0, latency_p99=750.0,
                error_rate=2.0, rps=350.0,
                extra_metrics={"db_queries_per_request": 6.8, "db_queries_baseline": 1.8,
                               "cache_miss_rate_percent": 65.0},
                logs=[
                    "2025-04-01T18:02:00Z WARN  search-service  Cache miss rate: 65% (baseline: 5%)",
                    "2025-04-01T18:04:00Z WARN  search-service  Query latency degraded",
                ],
                config={"db_pool_size": 20, "cache_ttl_sec": 120, "port": 8086},
                dependencies=["postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v1.5.0", "date": "2025-03-15", "status": "stable"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=30.0, memory=40.0, latency_p50=25.0, latency_p99=120.0,
                error_rate=0.3, rps=700.0,
                extra_metrics={"cache_miss_rate_percent": 60.0},
                logs=[
                    "2025-04-01T18:00:00Z INFO  user-service  Health check OK",
                    "2025-04-01T18:03:00Z WARN  user-service  Cache miss rate slightly elevated: 60%",
                ],
                config={"db_pool_size": 20, "cache_ttl_sec": 900, "port": 8082},
                dependencies=["postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "cache-service": ServiceDef(
                name="cache-service",
                # The trick: cache-service itself looks "healthy"
                cpu=20.0, memory=35.0, latency_p50=2.0, latency_p99=8.0,
                error_rate=0.0, rps=8000.0,
                extra_metrics={
                    "hit_rate_percent": 38.0,
                    "hit_rate_baseline_percent": 95.0,
                    "miss_rate_percent": 62.0,
                    "evictions_per_sec": 0,
                    "keys_total": 45000,
                    "keys_baseline": 120000,
                    "memory_used_percent": 15.0,
                },
                logs=[
                    "2025-04-01T16:30:00Z INFO  cache-service  Deployed v2.4.0",
                    "2025-04-01T16:30:05Z INFO  cache-service  Starting with new key hashing algorithm (MurmurHash3)",
                    "2025-04-01T16:30:10Z INFO  cache-service  Startup complete — all health checks passing",
                    "2025-04-01T17:00:00Z INFO  cache-service  Health check OK — 0 errors",
                    "2025-04-01T17:30:00Z INFO  cache-service  Health check OK — 0 errors",
                    "2025-04-01T18:00:00Z INFO  cache-service  Health check OK — 0 errors",
                    "2025-04-01T18:00:00Z DEBUG cache-service  Key distribution analysis: 62% of lookups returning MISS",
                    "2025-04-01T18:00:01Z DEBUG cache-service  Hash collision rate: 0.001% (normal)",
                    "2025-04-01T18:00:02Z DEBUG cache-service  NOTE: Key format changed in v2.4.0 — old keys not invalidated",
                ],
                config={
                    "hash_algorithm": "murmurhash3",
                    "hash_algorithm_previous": "fnv1a",
                    "max_memory": "4gb",
                    "key_format_version": "v2",
                    "port": 6380,
                },
                dependencies=[],
                deploy_history=[
                    {"version": "v2.3.0", "date": "2025-03-10", "status": "previous",
                     "notes": "Stable release, FNV1a hashing"},
                    {"version": "v2.4.0", "date": "2025-04-01T16:30:00Z", "status": "current",
                     "notes": "Switched key hashing from FNV1a to MurmurHash3 for better distribution. "
                              "NOTE: This changes the key format — existing cached entries will not be found."},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                cpu=78.0, memory=80.0, latency_p50=45.0, latency_p99=320.0,
                error_rate=0.5, rps=6500.0,
                extra_metrics={
                    "connections_active": 88, "connections_max": 100,
                    "queries_per_sec": 6500, "queries_baseline_per_sec": 1800,
                    "replication_lag_ms": 250,
                    "disk_io_percent": 85,
                },
                logs=[
                    "2025-04-01T18:00:00Z WARN  postgres  Connection count elevated (88/100)",
                    "2025-04-01T18:02:00Z WARN  postgres  Query throughput 3.6x above baseline",
                    "2025-04-01T18:03:00Z WARN  postgres  Disk I/O at 85% — consider scaling",
                    "2025-04-01T18:04:00Z WARN  postgres  Replication lag increased to 250ms",
                ],
                config={"max_connections": 100, "shared_buffers": "8GB"},
                dependencies=[],
                deploy_history=[],
            ),
        },
        alerts=[
            {"alert_id": "ALT-201", "severity": "warning",
             "service": "api-gateway",
             "message": "Aggregate p99 latency 4x above baseline (850ms vs 200ms)",
             "timestamp": "2025-04-01T18:05:00Z"},
            {"alert_id": "ALT-202", "severity": "warning",
             "service": "postgres-primary",
             "message": "DB connections at 88% capacity, query throughput 3.6x baseline",
             "timestamp": "2025-04-01T18:02:00Z"},
            {"alert_id": "ALT-203", "severity": "info",
             "service": "order-service",
             "message": "Cache miss rate elevated: 62% (baseline: 5%)",
             "timestamp": "2025-04-01T18:02:00Z"},
            {"alert_id": "ALT-204", "severity": "info",
             "service": "product-service",
             "message": "Cache miss rate elevated: 58%",
             "timestamp": "2025-04-01T18:01:00Z"},
        ],
        root_cause_service="cache-service",
        root_cause_keywords=["cache", "hash", "hashing", "key", "murmur", "fnv", "v2.4",
                             "deployment", "deploy", "miss", "format", "migration"],
        correct_remediation="rollback_deploy",
        correct_remediation_args={"service": "cache-service"},
        investigation_targets=["api-gateway", "order-service", "product-service",
                               "cache-service", "postgres-primary", "search-service"],
        penalty_services=[],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIUM-HARD — DNS Misconfiguration Causing Intermittent Failures
# ═══════════════════════════════════════════════════════════════════════════════

def build_dns_misconfiguration() -> Scenario:
    return Scenario(
        task_id="medium_dns_misconfiguration",
        task_name="DNS Misconfiguration Causing Intermittent Failures",
        difficulty="medium",
        goal=(
            "ALERT: order-service is experiencing intermittent failures when "
            "communicating with inventory-service. Some requests succeed while "
            "others timeout. Identify why the failures are intermittent and fix "
            "the underlying issue."
        ),
        description=(
            "A recent config change in order-service pointed it at a decommissioned "
            "internal DNS name (inventory-service-v2.internal) instead of the correct "
            "hostname (inventory-service.internal). DNS caching causes intermittent "
            "success/failure — cached entries resolve correctly until TTL expires."
        ),
        max_steps=15,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                cpu=35.0, memory=42.0, latency_p50=80.0, latency_p99=2200.0,
                error_rate=12.0, rps=900.0,
                logs=[
                    "2025-04-01T11:00:00Z INFO  api-gateway  Routing normal",
                    "2025-04-01T11:05:22Z WARN  api-gateway  Upstream order-service intermittent 502 errors",
                    "2025-04-01T11:06:10Z WARN  api-gateway  order-service error rate: 12%",
                    "2025-04-01T11:08:00Z INFO  api-gateway  order-service responding normally (temporary)",
                    "2025-04-01T11:10:30Z WARN  api-gateway  order-service 502 errors resumed",
                ],
                config={"timeout_ms": 5000, "retry_count": 2, "port": 8080},
                dependencies=["order-service", "user-service", "payment-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "order-service": ServiceDef(
                name="order-service",
                healthy=False,
                cpu=40.0, memory=55.0, latency_p50=45.0, latency_p99=8500.0,
                error_rate=28.0, rps=350.0,
                extra_metrics={
                    "dns_resolution_failures": 142,
                    "connection_timeouts": 98,
                    "successful_requests_percent": 72.0,
                    "inventory_calls_failed": 142,
                    "inventory_calls_total": 507,
                },
                logs=[
                    "2025-04-01T10:30:00Z INFO  order-service  Config reload triggered by deploy-bot",
                    "2025-04-01T10:30:01Z INFO  order-service  inventory_host changed: inventory-service.internal -> inventory-service-v2.internal",
                    "2025-04-01T10:35:15Z ERROR order-service  java.net.UnknownHostException: inventory-service-v2.internal: Name or service not known",
                    "2025-04-01T10:35:16Z WARN  order-service  Retrying inventory lookup... using cached DNS entry",
                    "2025-04-01T10:35:17Z INFO  order-service  Inventory call succeeded (cached DNS)",
                    "2025-04-01T10:40:22Z ERROR order-service  Connection to inventory-service-v2.internal timed out (5000ms)",
                    "2025-04-01T10:45:00Z ERROR order-service  java.net.UnknownHostException: inventory-service-v2.internal",
                    "2025-04-01T10:50:10Z WARN  order-service  28% of inventory requests failing — DNS resolution intermittent",
                    "2025-04-01T11:00:00Z ERROR order-service  Failed to place order: inventory check failed (DNS timeout)",
                ],
                config={
                    "inventory_host": "inventory-service-v2.internal",
                    "inventory_port": 8087,
                    "inventory_timeout_ms": 5000,
                    "dns_cache_ttl_sec": 30,
                    "db_pool_size": 30,
                    "port": 8083,
                    "config_last_modified": "2025-04-01T10:30:00Z",
                    "config_modified_by": "deploy-bot (service-mesh migration rule #12)",
                },
                dependencies=["inventory-service", "postgres-primary", "redis-cache"],
                deploy_history=[
                    {"version": "v4.8.0", "date": "2025-03-25", "status": "stable"},
                    {"version": "v4.8.0", "date": "2025-04-01", "status": "current",
                     "notes": "Config change: inventory_host updated for v2 service mesh migration"},
                ],
            ),
            "inventory-service": ServiceDef(
                name="inventory-service",
                cpu=22.0, memory=40.0, latency_p50=8.0, latency_p99=25.0,
                error_rate=0.1, rps=600.0,
                logs=[
                    "2025-04-01T11:00:00Z INFO  inventory-service  Health check OK",
                    "2025-04-01T11:01:00Z INFO  inventory-service  Processing 600 req/s — normal load",
                    "2025-04-01T11:02:00Z INFO  inventory-service  Note: incoming traffic from order-service appears reduced",
                ],
                config={
                    "hostname": "inventory-service.internal",
                    "port": 8087,
                    "db_pool_size": 20,
                },
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v2.5.0", "date": "2025-03-18", "status": "stable"},
                ],
            ),
            "payment-service": ServiceDef(
                name="payment-service",
                cpu=25.0, memory=38.0, latency_p50=12.0, latency_p99=40.0,
                error_rate=0.2, rps=400.0,
                logs=[
                    "2025-04-01T11:00:00Z INFO  payment-service  Health check OK",
                ],
                config={"port": 8081, "db_pool_size": 15},
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v3.2.1", "date": "2025-03-31", "status": "stable"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=20.0, memory=35.0, latency_p50=8.0, latency_p99=30.0,
                error_rate=0.05, rps=800.0,
                logs=[
                    "2025-04-01T11:00:00Z INFO  user-service  Health check OK",
                ],
                config={"db_pool_size": 15, "port": 8082},
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                cpu=30.0, memory=55.0, latency_p50=2.0, latency_p99=12.0,
                error_rate=0.0, rps=1800.0,
                extra_metrics={"connections_active": 35, "connections_max": 100},
                logs=[
                    "2025-04-01T11:00:00Z INFO  postgres  Connection count normal (35/100)",
                ],
                config={"max_connections": 100, "shared_buffers": "4GB"},
                dependencies=[],
                deploy_history=[],
            ),
            "redis-cache": ServiceDef(
                name="redis-cache",
                cpu=12.0, memory=45.0, latency_p50=1.0, latency_p99=4.0,
                error_rate=0.0, rps=4500.0,
                extra_metrics={"hit_rate_percent": 94.0},
                logs=[
                    "2025-04-01T11:00:00Z INFO  redis  Memory usage normal",
                ],
                config={"maxmemory": "2gb", "maxmemory-policy": "allkeys-lru"},
                dependencies=[],
                deploy_history=[],
            ),
        },
        alerts=[
            {"alert_id": "ALT-301", "severity": "critical",
             "service": "order-service",
             "message": "28% of requests failing — intermittent connection errors to inventory-service",
             "timestamp": "2025-04-01T11:00:00Z"},
            {"alert_id": "ALT-302", "severity": "warning",
             "service": "order-service",
             "message": "DNS resolution failures detected: 142 in last 30 minutes",
             "timestamp": "2025-04-01T11:00:00Z"},
            {"alert_id": "ALT-303", "severity": "warning",
             "service": "api-gateway",
             "message": "Elevated error rate from order-service (12%)",
             "timestamp": "2025-04-01T11:06:10Z"},
        ],
        root_cause_service="order-service",
        root_cause_keywords=["dns", "hostname", "inventory_host", "inventory-service-v2",
                             "config", "misconfiguration", "wrong", "host", "resolution"],
        correct_remediation="update_config",
        correct_remediation_args={"service": "order-service", "key": "inventory_host"},
        investigation_targets=["order-service", "inventory-service", "api-gateway"],
        penalty_services=["payment-service", "user-service", "postgres-primary", "redis-cache"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HARD — Database Replication Lag from Runaway Batch Job
# ═══════════════════════════════════════════════════════════════════════════════

def build_db_replication_lag() -> Scenario:
    return Scenario(
        task_id="hard_replication_lag",
        task_name="Database Replication Lag from Runaway Batch Job",
        difficulty="hard",
        goal=(
            "ALERT: Multiple services are returning stale data. Users report seeing "
            "outdated profiles, old inventory counts, and delayed order status updates. "
            "No service is down, but data consistency is degraded across the platform. "
            "Find the root cause and fix it."
        ),
        description=(
            "A batch analytics job was accidentally enabled on postgres-primary, causing "
            "heavy write amplification. This pushes replication lag on postgres-replica "
            "to 45+ seconds, so all services reading from the replica see stale data. "
            "The tricky part: all services appear healthy, latency is normal, and "
            "no errors are thrown — data is just stale."
        ),
        max_steps=20,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                cpu=30.0, memory=42.0, latency_p50=18.0, latency_p99=65.0,
                error_rate=0.1, rps=1100.0,
                logs=[
                    "2025-04-01T15:00:00Z INFO  api-gateway  Routing normal — no errors",
                    "2025-04-01T15:05:00Z INFO  api-gateway  All upstreams responding within SLA",
                    "2025-04-01T15:10:00Z INFO  api-gateway  No anomalies detected",
                ],
                config={"timeout_ms": 5000, "port": 8080},
                dependencies=["order-service", "user-service", "product-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=22.0, memory=38.0, latency_p50=10.0, latency_p99=35.0,
                error_rate=0.0, rps=800.0,
                extra_metrics={
                    "stale_reads_detected": 234,
                    "data_freshness_lag_sec": 47.0,
                    "read_source": "postgres-replica",
                },
                logs=[
                    "2025-04-01T15:00:00Z INFO  user-service  Health check OK",
                    "2025-04-01T15:02:00Z WARN  user-service  User profile read does not match recent write (stale read?)",
                    "2025-04-01T15:04:00Z WARN  user-service  234 stale-read incidents in last 15 minutes",
                    "2025-04-01T15:06:00Z WARN  user-service  Data freshness lag: 47s (threshold: 5s)",
                    "2025-04-01T15:08:00Z INFO  user-service  No application errors — reads returning successfully but data is outdated",
                ],
                config={
                    "db_read_host": "postgres-replica",
                    "db_write_host": "postgres-primary",
                    "db_pool_size": 15,
                    "port": 8082,
                },
                dependencies=["postgres-replica", "postgres-primary"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "order-service": ServiceDef(
                name="order-service",
                cpu=35.0, memory=50.0, latency_p50=15.0, latency_p99=55.0,
                error_rate=0.0, rps=500.0,
                extra_metrics={
                    "stale_reads_detected": 189,
                    "data_freshness_lag_sec": 45.0,
                    "read_source": "postgres-replica",
                    "order_status_mismatches": 89,
                },
                logs=[
                    "2025-04-01T15:00:00Z INFO  order-service  Processing orders normally",
                    "2025-04-01T15:03:00Z WARN  order-service  Order status mismatch: replica shows 'pending', primary shows 'shipped'",
                    "2025-04-01T15:05:00Z WARN  order-service  189 stale-read incidents detected",
                    "2025-04-01T15:07:00Z WARN  order-service  Read-after-write inconsistency: 45s lag",
                ],
                config={
                    "db_read_host": "postgres-replica",
                    "db_write_host": "postgres-primary",
                    "db_pool_size": 30,
                    "port": 8083,
                },
                dependencies=["postgres-replica", "postgres-primary", "redis-cache"],
                deploy_history=[
                    {"version": "v4.8.0", "date": "2025-03-25", "status": "stable"},
                ],
            ),
            "product-service": ServiceDef(
                name="product-service",
                cpu=28.0, memory=45.0, latency_p50=12.0, latency_p99=40.0,
                error_rate=0.0, rps=650.0,
                extra_metrics={
                    "stale_reads_detected": 156,
                    "data_freshness_lag_sec": 42.0,
                    "read_source": "postgres-replica",
                    "inventory_count_mismatches": 67,
                },
                logs=[
                    "2025-04-01T15:01:00Z WARN  product-service  Inventory count stale: showing 50 units, actual 12",
                    "2025-04-01T15:04:00Z WARN  product-service  156 stale-read incidents in last 15 min",
                    "2025-04-01T15:06:00Z INFO  product-service  No errors — data is readable but outdated",
                ],
                config={
                    "db_read_host": "postgres-replica",
                    "db_write_host": "postgres-primary",
                    "db_pool_size": 25,
                    "port": 8085,
                },
                dependencies=["postgres-replica", "postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v3.2.0", "date": "2025-03-22", "status": "stable"},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                healthy=False,
                cpu=92.0, memory=85.0, latency_p50=8.0, latency_p99=120.0,
                error_rate=0.0, rps=8500.0,
                extra_metrics={
                    "connections_active": 78, "connections_max": 100,
                    "write_ops_per_sec": 5200,
                    "write_ops_baseline": 800,
                    "replication_lag_sec": 47.0,
                    "replication_lag_baseline_sec": 0.1,
                    "wal_generation_rate_mb_sec": 45.0,
                    "wal_generation_baseline_mb_sec": 5.0,
                    "batch_job_write_ops": 4400,
                    "disk_io_percent": 92,
                },
                logs=[
                    "2025-04-01T14:00:00Z INFO  postgres-primary  Batch analytics job 'nightly_aggregation' started",
                    "2025-04-01T14:00:05Z INFO  postgres-primary  batch_job_enabled set to true by cron scheduler",
                    "2025-04-01T14:05:00Z WARN  postgres-primary  Write throughput 6.5x above baseline",
                    "2025-04-01T14:10:00Z WARN  postgres-primary  WAL generation rate: 45 MB/s (baseline: 5 MB/s)",
                    "2025-04-01T14:15:00Z WARN  postgres-primary  Replication lag increasing: 15s",
                    "2025-04-01T14:30:00Z WARN  postgres-primary  Replication lag: 32s — replica falling behind",
                    "2025-04-01T15:00:00Z WARN  postgres-primary  Replication lag: 47s — critical threshold exceeded",
                    "2025-04-01T15:00:01Z WARN  postgres-primary  Disk I/O at 92% — batch job consuming write bandwidth",
                ],
                config={
                    "max_connections": 100,
                    "shared_buffers": "8GB",
                    "batch_job_enabled": "true",
                    "batch_job_name": "nightly_aggregation",
                    "batch_job_started": "2025-04-01T14:00:00Z",
                    "batch_job_schedule": "daily 02:00 UTC (misconfigured to 14:00 UTC)",
                    "wal_level": "replica",
                },
                dependencies=[],
                deploy_history=[
                    {"version": "v15.4", "date": "2025-03-01", "status": "stable"},
                    {"version": "v15.4", "date": "2025-04-01", "status": "current",
                     "notes": "Cron schedule for nightly_aggregation changed: 02:00 UTC → 14:00 UTC (peak hours)"},
                ],
            ),
            "postgres-replica": ServiceDef(
                name="postgres-replica",
                cpu=65.0, memory=70.0, latency_p50=5.0, latency_p99=35.0,
                error_rate=0.0, rps=3500.0,
                extra_metrics={
                    "replication_lag_sec": 47.0,
                    "replication_lag_baseline_sec": 0.1,
                    "replay_rate_mb_sec": 12.0,
                    "wal_receiver_status": "streaming",
                    "bytes_behind_primary_mb": 1480,
                },
                logs=[
                    "2025-04-01T14:10:00Z WARN  postgres-replica  Replication lag: 15s",
                    "2025-04-01T14:30:00Z WARN  postgres-replica  Replication lag: 32s — falling behind primary",
                    "2025-04-01T15:00:00Z WARN  postgres-replica  Replication lag: 47s — 1480 MB behind primary",
                    "2025-04-01T15:00:01Z WARN  postgres-replica  WAL replay cannot keep up with primary write rate",
                    "2025-04-01T15:00:02Z INFO  postgres-replica  WAL receiver still streaming — not disconnected",
                ],
                config={
                    "primary_host": "postgres-primary",
                    "hot_standby": "on",
                    "max_standby_streaming_delay": "30s",
                },
                dependencies=["postgres-primary"],
                deploy_history=[],
            ),
            "redis-cache": ServiceDef(
                name="redis-cache",
                cpu=15.0, memory=48.0, latency_p50=1.0, latency_p99=5.0,
                error_rate=0.0, rps=5000.0,
                extra_metrics={"hit_rate_percent": 91.0},
                logs=[
                    "2025-04-01T15:00:00Z INFO  redis  Memory usage normal",
                ],
                config={"maxmemory": "2gb", "maxmemory-policy": "allkeys-lru"},
                dependencies=[],
                deploy_history=[],
            ),
            "cache-service": ServiceDef(
                name="cache-service",
                cpu=18.0, memory=32.0, latency_p50=2.0, latency_p99=7.0,
                error_rate=0.0, rps=6000.0,
                logs=[
                    "2025-04-01T15:00:00Z INFO  cache-service  Health check OK",
                ],
                config={"hash_algorithm": "fnv1a", "port": 6380},
                dependencies=[],
                deploy_history=[
                    {"version": "v2.3.0", "date": "2025-03-10", "status": "stable"},
                ],
            ),
        },
        alerts=[
            {"alert_id": "ALT-401", "severity": "warning",
             "service": "user-service",
             "message": "Stale data detected: 234 read inconsistencies in 15 minutes",
             "timestamp": "2025-04-01T15:04:00Z"},
            {"alert_id": "ALT-402", "severity": "warning",
             "service": "order-service",
             "message": "Order status mismatches: replica data 45s behind primary",
             "timestamp": "2025-04-01T15:05:00Z"},
            {"alert_id": "ALT-403", "severity": "warning",
             "service": "product-service",
             "message": "Inventory counts stale: 67 mismatches detected",
             "timestamp": "2025-04-01T15:04:00Z"},
            {"alert_id": "ALT-404", "severity": "info",
             "service": "postgres-primary",
             "message": "CPU at 92%, disk I/O at 92% — elevated write load",
             "timestamp": "2025-04-01T15:00:01Z"},
        ],
        root_cause_service="postgres-primary",
        root_cause_keywords=["batch", "job", "replication", "lag", "nightly", "aggregation",
                             "write", "wal", "schedule", "cron", "batch_job_enabled"],
        correct_remediation="update_config",
        correct_remediation_args={"service": "postgres-primary", "key": "batch_job_enabled"},
        investigation_targets=["user-service", "order-service", "product-service",
                               "postgres-primary", "postgres-replica"],
        penalty_services=["api-gateway", "redis-cache", "cache-service"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERT — Simultaneous Bad Deployment + Config Drift
# ═══════════════════════════════════════════════════════════════════════════════

def build_expert_multi_root_cause() -> Scenario:
    return Scenario(
        task_id="expert_multi_root_cause",
        task_name="Simultaneous Bad Deployment and Config Drift",
        difficulty="expert",
        goal=(
            "ALERT: System-wide degradation with two distinct failure patterns. "
            "The search-service is returning 500 errors after a deployment, AND "
            "the order-service is experiencing DB connection timeouts from a config change. "
            "These are independent issues happening simultaneously. Identify and fix BOTH "
            "root causes."
        ),
        description=(
            "Two things broke at the same time: (1) search-service deployed v3.1.0 with a "
            "broken Elasticsearch query builder, causing 500 errors, and (2) a config bot "
            "reduced order-service's DB connection pool from 50 to 3, causing timeouts. "
            "The agent must identify that there are TWO separate issues and fix both. "
            "Fixing only one leaves the system partially broken."
        ),
        max_steps=25,
        services={
            "api-gateway": ServiceDef(
                name="api-gateway",
                healthy=False,
                cpu=65.0, memory=55.0, latency_p50=450.0, latency_p99=6800.0,
                error_rate=28.0, rps=600.0,
                logs=[
                    "2025-04-01T12:00:00Z ERROR api-gateway  High error rate: 28% of requests failing",
                    "2025-04-01T12:00:10Z WARN  api-gateway  Upstream search-service returning 500s",
                    "2025-04-01T12:00:15Z WARN  api-gateway  Upstream order-service timeout (5000ms)",
                    "2025-04-01T12:00:20Z ERROR api-gateway  Two distinct failure patterns detected",
                    "2025-04-01T12:01:00Z WARN  api-gateway  Circuit breaker OPEN for search-service",
                    "2025-04-01T12:01:30Z WARN  api-gateway  Circuit breaker OPEN for order-service",
                ],
                config={"timeout_ms": 5000, "circuit_breaker_threshold": 5, "port": 8080},
                dependencies=["search-service", "order-service", "user-service", "product-service"],
                deploy_history=[
                    {"version": "v2.14.0", "date": "2025-03-20", "status": "stable"},
                ],
            ),
            "search-service": ServiceDef(
                name="search-service",
                healthy=False,
                cpu=30.0, memory=45.0, latency_p50=5.0, latency_p99=15.0,
                error_rate=95.0, rps=200.0,
                extra_metrics={
                    "http_500_count_last_hour": 4280,
                    "elasticsearch_query_errors": 4280,
                    "successful_searches": 220,
                },
                logs=[
                    "2025-04-01T11:45:00Z INFO  search-service  Deployed v3.1.0",
                    "2025-04-01T11:45:05Z INFO  search-service  Starting with updated Elasticsearch query builder",
                    "2025-04-01T11:45:10Z ERROR search-service  ElasticsearchParseException: unknown query type [match_phrase_prefix_v2]",
                    "2025-04-01T11:45:11Z ERROR search-service  Failed to build search query: unsupported query DSL",
                    "2025-04-01T11:50:00Z ERROR search-service  4280 query failures in last 5 minutes",
                    "2025-04-01T12:00:00Z ERROR search-service  Error rate at 95% — all full-text searches failing",
                    "2025-04-01T12:00:01Z INFO  search-service  Simple ID lookups still working (5% of traffic)",
                ],
                config={
                    "elasticsearch_host": "elasticsearch.internal",
                    "query_builder_version": "v2",
                    "port": 8086,
                    "max_results": 100,
                },
                dependencies=["elasticsearch"],
                deploy_history=[
                    {"version": "v3.0.2", "date": "2025-03-20", "status": "previous",
                     "notes": "Stable release with original query builder"},
                    {"version": "v3.1.0", "date": "2025-04-01T11:45:00Z", "status": "current",
                     "notes": "Updated Elasticsearch query builder to v2 DSL format. "
                              "WARNING: requires Elasticsearch 8.x (cluster is on 7.x)"},
                ],
            ),
            "order-service": ServiceDef(
                name="order-service",
                healthy=False,
                cpu=85.0, memory=72.0, latency_p50=5200.0, latency_p99=9900.0,
                error_rate=45.0, rps=40.0,
                extra_metrics={
                    "db_pool_active": 3, "db_pool_max": 3,
                    "db_pool_pending_requests": 245,
                    "thread_pool_active": 198, "thread_pool_max": 200,
                },
                logs=[
                    "2025-04-01T11:30:00Z INFO  order-service  Config reload triggered",
                    "2025-04-01T11:30:01Z INFO  order-service  db_pool_size changed: 50 -> 3",
                    "2025-04-01T11:35:00Z WARN  order-service  Connection pool exhausted — requests queuing",
                    "2025-04-01T11:45:00Z ERROR order-service  DB connection timeout (30s) — 245 requests waiting",
                    "2025-04-01T12:00:00Z ERROR order-service  Thread starvation: 198/200 threads blocked on DB pool",
                    "2025-04-01T12:00:05Z ERROR order-service  org.postgresql.util.PSQLException: Cannot get connection",
                ],
                config={
                    "db_pool_size": 3,
                    "db_pool_max_wait_ms": 30000,
                    "port": 8083,
                    "thread_pool_size": 200,
                    "config_last_modified": "2025-04-01T11:30:00Z",
                    "config_modified_by": "deploy-bot (capacity-planner rule #8)",
                },
                dependencies=["postgres-primary", "redis-cache"],
                deploy_history=[
                    {"version": "v4.8.0", "date": "2025-03-25", "status": "stable"},
                    {"version": "v4.8.0", "date": "2025-04-01", "status": "current",
                     "notes": "Config change: db_pool_size adjusted by capacity-planner"},
                ],
            ),
            "user-service": ServiceDef(
                name="user-service",
                cpu=22.0, memory=36.0, latency_p50=9.0, latency_p99=32.0,
                error_rate=0.1, rps=700.0,
                logs=["2025-04-01T12:00:00Z INFO  user-service  Health check OK"],
                config={"db_pool_size": 20, "port": 8082},
                dependencies=["postgres-primary"],
                deploy_history=[
                    {"version": "v5.0.2", "date": "2025-03-10", "status": "stable"},
                ],
            ),
            "product-service": ServiceDef(
                name="product-service",
                cpu=25.0, memory=40.0, latency_p50=12.0, latency_p99=45.0,
                error_rate=0.2, rps=500.0,
                logs=["2025-04-01T12:00:00Z INFO  product-service  Health check OK"],
                config={"db_pool_size": 25, "port": 8085},
                dependencies=["postgres-primary", "cache-service"],
                deploy_history=[
                    {"version": "v3.2.0", "date": "2025-03-22", "status": "stable"},
                ],
            ),
            "postgres-primary": ServiceDef(
                name="postgres-primary",
                cpu=45.0, memory=65.0, latency_p50=3.0, latency_p99=20.0,
                error_rate=0.0, rps=2000.0,
                extra_metrics={"connections_active": 48, "connections_max": 100},
                logs=["2025-04-01T12:00:00Z INFO  postgres  Connection count normal (48/100)"],
                config={"max_connections": 100, "shared_buffers": "8GB"},
                dependencies=[],
                deploy_history=[],
            ),
            "redis-cache": ServiceDef(
                name="redis-cache",
                cpu=12.0, memory=42.0, latency_p50=1.0, latency_p99=4.0,
                error_rate=0.0, rps=4500.0,
                logs=["2025-04-01T12:00:00Z INFO  redis  Memory usage normal"],
                config={"maxmemory": "2gb"},
                dependencies=[],
                deploy_history=[],
            ),
            "elasticsearch": ServiceDef(
                name="elasticsearch",
                cpu=35.0, memory=70.0, latency_p50=5.0, latency_p99=25.0,
                error_rate=0.0, rps=3000.0,
                extra_metrics={"cluster_status": "green", "version": "7.17.8"},
                logs=[
                    "2025-04-01T12:00:00Z INFO  elasticsearch  Cluster health: green",
                    "2025-04-01T12:00:05Z WARN  elasticsearch  Rejecting queries with unknown type 'match_phrase_prefix_v2'",
                    "2025-04-01T12:00:06Z INFO  elasticsearch  Cluster version: 7.17.8 (v2 DSL requires 8.x)",
                ],
                config={"version": "7.17.8", "cluster_name": "prod-search"},
                dependencies=[],
                deploy_history=[
                    {"version": "7.17.8", "date": "2025-02-15", "status": "stable"},
                ],
            ),
        },
        alerts=[
            {"alert_id": "ALT-501", "severity": "critical",
             "service": "search-service",
             "message": "95% error rate — all full-text searches returning 500",
             "timestamp": "2025-04-01T12:00:00Z"},
            {"alert_id": "ALT-502", "severity": "critical",
             "service": "order-service",
             "message": "DB connection pool exhausted — 245 requests queued (pool size: 3)",
             "timestamp": "2025-04-01T12:00:00Z"},
            {"alert_id": "ALT-503", "severity": "critical",
             "service": "api-gateway",
             "message": "Error rate at 28% — two circuit breakers OPEN",
             "timestamp": "2025-04-01T12:00:20Z"},
            {"alert_id": "ALT-504", "severity": "warning",
             "service": "api-gateway",
             "message": "Two distinct failure patterns: search 500s AND order timeouts",
             "timestamp": "2025-04-01T12:01:00Z"},
        ],
        # For multi-root-cause: primary root cause is search-service (deploy)
        root_cause_service="search-service",
        root_cause_keywords=["search", "deployment", "deploy", "elasticsearch", "query",
                             "rollback", "order", "pool", "connection", "config",
                             "db_pool_size", "both", "two", "multiple"],
        correct_remediation="multi",  # special: requires multiple fixes
        correct_remediations=[
            {"cmd": "rollback_deploy", "service": "search-service"},
            {"cmd": "update_config", "service": "order-service", "key": "db_pool_size"},
        ],
        correct_remediation_args={},
        investigation_targets=["api-gateway", "search-service", "order-service",
                               "elasticsearch"],
        penalty_services=["user-service", "product-service", "postgres-primary", "redis-cache"],
    )


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_SCENARIOS: dict[str, Scenario] = {}


def _register():
    global ALL_SCENARIOS
    for builder in (build_easy, build_medium, build_hard,
                    build_dns_misconfiguration, build_db_replication_lag,
                    build_expert_multi_root_cause):
        s = builder()
        ALL_SCENARIOS[s.task_id] = s


_register()
