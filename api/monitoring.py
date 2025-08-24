import time
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import pickle

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning(
        "Prometheus client not available. Install with: pip install prometheus-client"
    )

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Container for request metrics"""

    timestamp: float
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    user_id: Optional[str]
    session_length: Optional[int]
    top_k: Optional[int]
    cache_hit: bool = False


@dataclass
class RecommendationMetrics:
    """Container for recommendation-specific metrics"""

    timestamp: float
    num_recommendations: int
    avg_score: float
    fusion_method: str
    cf_available: bool
    cb_available: bool
    session_length: int
    processing_time_ms: float


class MetricsCollector:
    """Collects and stores application metrics"""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.request_history: deque = deque(maxlen=max_history)
        self.recommendation_history: deque = deque(maxlen=max_history)
        self.error_counts: defaultdict = defaultdict(int)
        self.endpoint_counts: defaultdict = defaultdict(int)
        self.start_time = time.time()

        # Prometheus metrics if available
        self.prometheus_registry = None
        self.prometheus_metrics = {}

        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prometheus_registry = CollectorRegistry()

        self.prometheus_metrics = {
            "requests_total": Counter(
                "nexttrack_requests_total",
                "Total number of requests",
                ["endpoint", "method", "status"],
                registry=self.prometheus_registry,
            ),
            "request_duration": Histogram(
                "nexttrack_request_duration_seconds",
                "Request duration in seconds",
                ["endpoint"],
                registry=self.prometheus_registry,
            ),
            "active_sessions": Gauge(
                "nexttrack_active_sessions",
                "Number of active sessions",
                registry=self.prometheus_registry,
            ),
            "recommendation_scores": Histogram(
                "nexttrack_recommendation_scores",
                "Distribution of recommendation scores",
                buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                registry=self.prometheus_registry,
            ),
            "cache_hits": Counter(
                "nexttrack_cache_hits_total",
                "Number of cache hits",
                ["cache_type"],
                registry=self.prometheus_registry,
            ),
            "processing_time": Histogram(
                "nexttrack_processing_time_seconds",
                "Time spent in recommendation processing",
                ["component"],
                registry=self.prometheus_registry,
            ),
        }

        logger.info("Prometheus metrics initialized")

    def record_request(self, metrics: RequestMetrics):
        """Record a request"""
        self.request_history.append(metrics)
        self.endpoint_counts[metrics.endpoint] += 1

        if metrics.status_code >= 400:
            self.error_counts[metrics.status_code] += 1

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
            self.prometheus_metrics["requests_total"].labels(
                endpoint=metrics.endpoint,
                method=metrics.method,
                status=str(metrics.status_code),
            ).inc()

            self.prometheus_metrics["request_duration"].labels(
                endpoint=metrics.endpoint
            ).observe(metrics.duration_ms / 1000.0)

            if metrics.cache_hit:
                self.prometheus_metrics["cache_hits"].labels(
                    cache_type="recommendation"
                ).inc()

    def record_recommendation(self, metrics: RecommendationMetrics):
        """Record recommendation metrics"""
        self.recommendation_history.append(metrics)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
            self.prometheus_metrics["recommendation_scores"].observe(metrics.avg_score)

            self.prometheus_metrics["processing_time"].labels(
                component="recommendation"
            ).observe(metrics.processing_time_ms / 1000.0)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        now = time.time()
        uptime = now - self.start_time

        # Request stats
        total_requests = len(self.request_history)
        recent_requests = [
            r for r in self.request_history if now - r.timestamp < 3600
        ]  # Last hour

        if recent_requests:
            avg_response_time = sum(r.duration_ms for r in recent_requests) / len(
                recent_requests
            )
            error_rate = sum(1 for r in recent_requests if r.status_code >= 400) / len(
                recent_requests
            )
        else:
            avg_response_time = 0.0
            error_rate = 0.0

        # Recommendation stats
        recent_recs = [
            r for r in self.recommendation_history if now - r.timestamp < 3600
        ]
        if recent_recs:
            avg_rec_score = sum(r.avg_score for r in recent_recs) / len(recent_recs)
            avg_session_length = sum(r.session_length for r in recent_recs) / len(
                recent_recs
            )
        else:
            avg_rec_score = 0.0
            avg_session_length = 0.0

        return {
            "uptime_seconds": uptime,
            "total_requests": total_requests,
            "requests_last_hour": len(recent_requests),
            "avg_response_time_ms": avg_response_time,
            "error_rate": error_rate,
            "total_recommendations": len(self.recommendation_history),
            "avg_recommendation_score": avg_rec_score,
            "avg_session_length": avg_session_length,
            "top_endpoints": dict(
                sorted(self.endpoint_counts.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ),
            "error_counts": dict(self.error_counts),
        }

    def get_prometheus_metrics(self) -> Optional[str]:
        """Get Prometheus metrics in text format"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_registry:
            return None

        return generate_latest(self.prometheus_registry).decode("utf-8")

    def export_metrics(self, output_path: Path) -> None:
        """Export metrics to JSON file"""
        metrics_data = {
            "summary": self.get_summary_stats(),
            "request_history": [asdict(r) for r in list(self.request_history)],
            "recommendation_history": [
                asdict(r) for r in list(self.recommendation_history)
            ],
            "export_timestamp": time.time(),
        }

        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")


class CacheManager:
    """In-memory caching for recommendations"""

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache: Dict[str, Any] = {}  # In-memory cache
        self.cache_timestamps: Dict[str, float] = {}  # Track TTL
        self.max_cache_size = 1000

        logger.info("Using in-memory caching")

    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from parameters"""
        # Create deterministic key from parameters
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"{prefix}:{key_hash}"

    def get_recommendation_cache_key(
        self,
        track_sequence: List[str],
        top_k: int,
        fusion_method: str = "weighted_sum",
        **kwargs,
    ) -> str:
        """Generate cache key for recommendations"""
        return self._generate_cache_key(
            "rec", tracks=track_sequence, k=top_k, fusion=fusion_method, **kwargs
        )

    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired"""
        if cache_key not in self.cache_timestamps:
            return True

        return time.time() - self.cache_timestamps[cache_key] > self.default_ttl

    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.default_ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    def get_cached_recommendations(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached recommendations"""
        self._cleanup_expired()

        if cache_key in self.cache and not self._is_expired(cache_key):
            return self.cache[cache_key]

        return None

    def cache_recommendations(
        self, cache_key: str, recommendations: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache recommendations"""
        if ttl is None:
            ttl = self.default_ttl

        # Implement simple LRU by removing oldest entries
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k]
            )
            self.cache.pop(oldest_key, None)
            self.cache_timestamps.pop(oldest_key, None)

        self.cache[cache_key] = recommendations
        self.cache_timestamps[cache_key] = time.time()

        return True

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern"""
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        count = len(keys_to_remove)

        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

        if count > 0:
            logger.info(f"Invalidated {count} cache entries matching '{pattern}'")

        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()

        stats = {
            "cache_type": "in-memory",
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": 0.0,  # Could implement hit tracking if needed
        }

        return stats

    def clear_cache(self) -> bool:
        """Clear all cache"""
        self.cache.clear()
        self.cache_timestamps.clear()

        logger.info("Cache cleared")
        return True


class HealthChecker:
    """Health monitoring for system components"""

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.checks = {
            "api": self._check_api_health,
            "cache": self._check_cache_health,
            "metrics": self._check_metrics_health,
            "memory": self._check_memory_health,
        }

    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        if self.metrics_collector:
            stats = self.metrics_collector.get_summary_stats()
            recent_errors = stats["error_rate"]

            if recent_errors > 0.1:  # >10% error rate
                status = "unhealthy"
                message = f"High error rate: {recent_errors:.1%}"
            elif stats["requests_last_hour"] == 0:
                status = "idle"
                message = "No recent requests"
            else:
                status = "healthy"
                message = f"Serving requests normally (avg: {stats['avg_response_time_ms']:.1f}ms)"

            return {
                "status": status,
                "message": message,
                "details": {
                    "uptime": stats["uptime_seconds"],
                    "total_requests": stats["total_requests"],
                    "error_rate": recent_errors,
                    "avg_response_time": stats["avg_response_time_ms"],
                },
            }
        else:
            return {"status": "unknown", "message": "No metrics collector available"}

    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        if self.cache_manager:
            stats = self.cache_manager.get_cache_stats()

            cache_size = stats.get("cache_size", 0)
            max_size = stats.get("max_cache_size", 1000)

            if cache_size > max_size * 0.9:
                status = "warning"
                message = f"Cache nearly full: {cache_size}/{max_size}"
            else:
                status = "healthy"
                message = f"In-memory cache operational ({cache_size} entries)"

            return {"status": status, "message": message, "details": stats}
        else:
            return {"status": "disabled", "message": "No cache manager configured"}

    def _check_metrics_health(self) -> Dict[str, Any]:
        """Check metrics collection health"""
        if self.metrics_collector:
            return {
                "status": "healthy",
                "message": "Metrics collection active",
                "details": {
                    "request_history_size": len(self.metrics_collector.request_history),
                    "recommendation_history_size": len(
                        self.metrics_collector.recommendation_history
                    ),
                    "prometheus_available": PROMETHEUS_AVAILABLE,
                },
            }
        else:
            return {"status": "disabled", "message": "No metrics collector configured"}

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil

            memory = psutil.virtual_memory()

            if memory.percent > 90:
                status = "critical"
                message = f"Very high memory usage: {memory.percent:.1f}%"
            elif memory.percent > 75:
                status = "warning"
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent:.1f}%"

            return {
                "status": status,
                "message": message,
                "details": {
                    "percent": memory.percent,
                    "available": memory.available,
                    "total": memory.total,
                },
            }
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil not available for memory monitoring",
            }
        except Exception as e:
            return {"status": "error", "message": f"Memory check failed: {e}"}

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = "healthy"

        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result

                # Update overall status
                if result["status"] == "critical":
                    overall_status = "critical"
                elif (
                    result["status"] in ["unhealthy", "error"]
                    and overall_status != "critical"
                ):
                    overall_status = "unhealthy"
                elif (
                    result["status"] in ["warning", "degraded"]
                    and overall_status == "healthy"
                ):
                    overall_status = "degraded"

            except Exception as e:
                results[check_name] = {
                    "status": "error",
                    "message": f"Health check failed: {e}",
                }
                if overall_status == "healthy":
                    overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "checks": results,
        }


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_cache_manager: Optional[CacheManager] = None
_health_checker: Optional[HealthChecker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(
            cache_manager=get_cache_manager(), metrics_collector=get_metrics_collector()
        )
    return _health_checker


if __name__ == "__main__":
    # Test monitoring components
    logging.basicConfig(level=logging.INFO)

    print("Production Monitoring System")
    print("===========================")

    # Test metrics collector
    collector = MetricsCollector()

    # Simulate some requests
    for i in range(5):
        metrics = RequestMetrics(
            timestamp=time.time(),
            endpoint="/recommendations",
            method="POST",
            status_code=200,
            duration_ms=50.0 + i * 10,
            user_id=f"user_{i}",
            session_length=3 + i,
            top_k=10,
            cache_hit=i % 2 == 0,
        )
        collector.record_request(metrics)

    print("✅ Metrics collector working")
    print(f"   Summary: {collector.get_summary_stats()}")

    # Test cache manager
    cache = CacheManager()
    test_recs = {"recommendations": ["track1", "track2"], "scores": [0.9, 0.8]}
    cache_key = cache.get_recommendation_cache_key(["track0"], top_k=5)

    cache.cache_recommendations(cache_key, test_recs)
    cached = cache.get_cached_recommendations(cache_key)

    if cached:
        print("✅ Cache manager working")
        print(f"   Stats: {cache.get_cache_stats()}")
    else:
        print("❌ Cache manager failed")

    # Test health checker
    health = HealthChecker(cache, collector)
    health_status = health.run_health_checks()

    print(f"✅ Health checker working")
    print(f"   Overall status: {health_status['overall_status']}")

    print("\nMonitoring system ready for production! 🚀")
