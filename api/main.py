from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import os

from api.hybrid_recommender import HybridRecommender
from api.monitoring import (
    get_metrics_collector,
    get_cache_manager,
    get_health_checker,
    RequestMetrics,
    RecommendationMetrics,
)
from api.ab_testing import (
    get_feedback_collector,
    get_ab_tester,
    FeedbackEvent,
    FeedbackType,
    ExperimentalRecommender,
)
from config.settings import settings

print("api/main.py module loaded")

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL), format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global recommender instance
recommender: Optional[HybridRecommender] = None
experimental_recommender: Optional[ExperimentalRecommender] = None

# Rate limiting storage
rate_limit_storage = defaultdict(lambda: deque())

# Security
security = HTTPBearer() if settings.ENABLE_AUTH else None


def get_api_key_dependency():
    if settings.ENABLE_AUTH:
        return Security(security)
    else:

        async def _noop():
            return "anonymous"

        return _noop


class RateLimiter:
    """Simple in-memory rate limiter"""

    @staticmethod
    def is_rate_limited(client_ip: str) -> bool:
        now = time.time()
        window_start = now - settings.RATE_LIMIT_WINDOW
        client_requests = rate_limit_storage[client_ip]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        if len(client_requests) >= settings.RATE_LIMIT_REQUESTS:
            return True
        client_requests.append(now)
        return False


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(get_api_key_dependency()),
) -> str:
    """Verify API key if authentication is enabled"""
    if not settings.ENABLE_AUTH:
        return "anonymous"

    if not credentials or credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "authenticated"


async def check_rate_limit(request: Request) -> None:
    """Check rate limiting"""
    client_ip = request.client.host

    if RateLimiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI lifespan starting")
    """Application lifespan events"""
    # Startup
    global recommender, experimental_recommender
    logger.info("NextTrack API is starting up...")

    try:
        # Initialize hybrid recommender
        recommender = HybridRecommender(
            cf_weight=settings.CF_WEIGHT, cb_weight=settings.CB_WEIGHT
        )

        # Determine content features path
        content_features_path = None
        if settings.USE_CONTENT_BASED:
            if settings.METADATA_FEATURES_FILE.exists():
                content_features_path = settings.METADATA_FEATURES_FILE

        # Initialize recommender
        index_type = (
            "faiss"
            if settings.USE_ANN_INDEX and settings.ANN_INDEX_TYPE == "faiss"
            else "brute_force"
        )

        recommender.initialize(
            embeddings_path=settings.EMBEDDINGS_FILE,
            cf_index_type=index_type,
            use_content_based=settings.USE_CONTENT_BASED,
            content_features_path=content_features_path,
        )

        print(f"EMBEDDINGS_FILE resolved to: {settings.EMBEDDINGS_FILE}")
        print(f"METADATA_FEATURES_FILE resolved to: {settings.METADATA_FEATURES_FILE}")
        logger.info(f"API running at http://{settings.HOST}:{settings.PORT}")

        # Initialize experimental recommender
        experimental_recommender = ExperimentalRecommender(
            base_recommender=recommender, ab_tester=get_ab_tester()
        )

        logger.info("NextTrack API startup complete")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    logger.info("NextTrack API is shutting down...")


# Create FastAPI app with comprehensive OpenAPI documentation
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
## NextTrack Hybrid Music Recommendation API

A production-ready hybrid music recommendation system that combines collaborative filtering (CF) 
and content-based (CB) approaches to provide personalized music recommendations.

### Key Features

* **🎵 Hybrid Recommendations**: Combines collaborative filtering and content-based approaches
* **⚡ Real-time API**: Sub-millisecond response times with FAISS indexing
* **🧪 A/B Testing**: Built-in experimentation framework for optimization
* **📊 User Feedback**: Comprehensive feedback collection and analysis
* **🔒 Security**: Optional API key authentication and rate limiting
* **📈 Monitoring**: Prometheus metrics and health checks
* **🐳 Docker Ready**: Production containerization

### Architecture

The system uses a modular architecture with:
- **CF Engine**: LightFM-based collaborative filtering with ANN indexing
- **CB Engine**: Content-based filtering using audio/metadata features  
- **Hybrid Layer**: Configurable fusion of CF and CB scores
- **Session Vectorizer**: Advanced session encoding (mean pooling or learned)
- **A/B Testing**: Experiment management and user assignment

### Getting Started

1. **Authentication**: Include `Authorization: Bearer your-api-key` header if auth is enabled
2. **Rate Limits**: Default 100 requests per hour per IP
3. **Recommendation Flow**: 
   - Submit listening history via `/recommendations`
   - Get personalized recommendations
   - Provide feedback via `/feedback` endpoints
4. **Monitoring**: Check `/health` for system status

### Support

- **Documentation**: Comprehensive API docs and examples
- **Health Check**: `/health` endpoint for system monitoring
- **Metrics**: `/metrics` endpoint for Prometheus integration
""",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "NextTrack API Support",
        "email": "support@nexttrack.ai",
        "url": "https://github.com/nexttrack/api",
    },
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    servers=[
        {
            "url": f"http://{settings.HOST}:{settings.PORT}",
            "description": "Development server",
        },
        {"url": "https://api.nexttrack.ai", "description": "Production server"},
    ],
    tags_metadata=[
        {"name": "Health", "description": "System health and status endpoints"},
        {
            "name": "Recommendations",
            "description": "Core music recommendation functionality",
        },
        {"name": "Feedback", "description": "User feedback collection and analytics"},
        {"name": "Experiments", "description": "A/B testing and experiment management"},
        {
            "name": "Monitoring",
            "description": "System monitoring, metrics, and cache management",
        },
        {"name": "System", "description": "System configuration and statistics"},
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class RecommendationRequest(BaseModel):
    track_sequence: List[str] = Field(
        ..., min_length=1, description="User's listening history"
    )
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K)
    genre_filter: Optional[List[str]] = Field(
        default=None, description="Optional genre constraints"
    )
    mood_filter: Optional[List[str]] = Field(
        default=None, description="Optional mood constraints"
    )
    fusion_method: str = Field(
        default="weighted_sum", description="Fusion method for hybrid scores"
    )

    @field_validator("fusion_method")
    @classmethod
    def validate_fusion_method(cls, v):
        if v not in ["weighted_sum", "max", "rank_fusion"]:
            raise ValueError(
                "fusion_method must be one of: weighted_sum, max, rank_fusion"
            )
        return v


class RecommendationResponse(BaseModel):
    recommendations: List[str] = Field(..., description="List of recommended track IDs")
    scores: List[float] = Field(
        ..., description="Confidence scores for each recommendation"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Additional recommendation metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "recommendations": ["track_123", "track_456", "track_789"],
                "scores": [0.95, 0.87, 0.82],
                "metadata": {
                    "cf_weight": 0.7,
                    "cb_weight": 0.3,
                    "session_length": 5,
                    "fusion_method": "weighted_sum",
                    "processing_time_ms": 15.4,
                },
            }
        }


class SimilarTracksRequest(BaseModel):
    track_id: str = Field(..., description="Track ID to find similarities for")
    top_k: int = Field(default=10, ge=1, le=settings.MAX_TOP_K)
    use_cf: bool = Field(default=True, description="Use collaborative filtering")
    use_cb: bool = Field(default=True, description="Use content-based filtering")


class SimilarTracksResponse(BaseModel):
    track_id: str = Field(..., description="Original track ID")
    similar_tracks: List[str] = Field(..., description="List of similar track IDs")
    scores: List[float] = Field(..., description="Similarity scores")
    method: str = Field(..., description="Method used for similarity calculation")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(..., description="Health check timestamp")
    components: Dict[str, bool] = Field(..., description="Component status map")


class FeedbackResponse(BaseModel):
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Success message")
    timestamp: float = Field(..., description="Operation timestamp")


class FeedbackStatsResponse(BaseModel):
    feedback_stats: Dict[str, Any] = Field(
        ..., description="Aggregated feedback statistics"
    )
    timestamp: float = Field(..., description="Stats generation timestamp")


class ExperimentResponse(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier")
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Operation message")


class ExperimentListResponse(BaseModel):
    experiments: List[Dict[str, Any]] = Field(..., description="List of experiments")
    total_count: int = Field(..., description="Total number of experiments")


class SystemStatsResponse(BaseModel):
    status: str
    weights: Optional[Dict[str, float]] = None
    components: Optional[Dict[str, bool]] = None
    cf_stats: Optional[Dict[str, Any]] = None
    cb_stats: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: Optional[float] = Field(default=None, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Validation error description",
                "status_code": 422,
                "timestamp": 1692902400.0,
            }
        }


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    user_id: Optional[str] = Field(default=None, description="User ID (optional)")
    track_id: str = Field(..., description="Track ID that received feedback")
    feedback_type: str = Field(
        ..., description="Type of feedback: like, dislike, skip, play_complete"
    )
    feedback_value: Optional[float] = Field(
        default=None, description="Feedback value (e.g., rating 1-5, play duration)"
    )
    recommendation_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Context when recommendation was made"
    )

    @field_validator("feedback_type")
    @classmethod
    def validate_feedback_type(cls, v):
        valid_types = ["like", "dislike", "skip", "play_complete", "explicit_rating"]
        if v not in valid_types:
            raise ValueError(f"feedback_type must be one of: {valid_types}")
        return v


class FeedbackStatsResponse(BaseModel):
    feedback_stats: Dict[str, Any] = Field(
        ..., description="Aggregated feedback statistics"
    )
    timestamp: float = Field(..., description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "feedback_stats": {
                    "total_events": 1250,
                    "feedback_by_type": {
                        "like": 450,
                        "dislike": 123,
                        "skip": 567,
                        "play_complete": 110,
                    },
                    "engagement_rate": 0.76,
                    "satisfaction_score": 3.8,
                },
                "timestamp": 1692902400.0,
            }
        }


class ExperimentsListResponse(BaseModel):
    experiments: List[Dict[str, Any]] = Field(
        ..., description="List of active experiments"
    )
    total_count: int = Field(..., description="Total number of experiments")
    timestamp: float = Field(..., description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "experiments": [
                    {
                        "experiment_id": "exp_1692902400",
                        "name": "CF Weight Optimization",
                        "description": "Testing different collaborative filtering weights",
                        "status": "active",
                        "traffic_split": 0.2,
                        "start_date": "2023-08-24T20:00:00",
                        "end_date": "2023-09-24T20:00:00",
                        "success_metrics": ["precision@10", "engagement_rate"],
                    }
                ],
                "total_count": 1,
                "timestamp": 1692902400.0,
            }
        }


class ExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    traffic_split: float = Field(
        ..., ge=0.0, le=1.0, description="Traffic percentage (0.0-1.0)"
    )
    config_overrides: Dict[str, Any] = Field(..., description="Configuration overrides")
    success_metrics: List[str] = Field(
        default=["ctr", "engagement"], description="Success metrics to track"
    )
    duration_days: int = Field(
        default=30, ge=1, le=365, description="Experiment duration in days"
    )


# API Endpoints
@app.get(
    "/",
    tags=["Health"],
    summary="Basic Health Check",
    description="Simple health check endpoint to verify the API is running",
    response_description="Basic status information",
)
async def root():
    """Health check endpoint"""
    return {
        "message": "NextTrack API is running and healthy!",
        "version": settings.API_VERSION,
        "status": "ok",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Detailed Health Check",
    description="""
    Comprehensive health check that verifies all system components are properly initialized.
    
    **Returns:**
    - System status
    - Component availability
    - Timestamp
    - Configuration status
    """,
    response_model=HealthResponse,
    response_description="Detailed system health information",
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": 1692902400.0,
                        "components": {
                            "recommender": True,
                            "auth_enabled": False,
                            "rate_limiting": True,
                        },
                    }
                }
            },
        },
        503: {
            "description": "System is unhealthy",
            "content": {
                "application/json": {
                    "example": {"detail": "Recommender system not initialized"}
                }
            },
        },
    },
)
async def health_check():
    """Detailed health check"""
    global recommender

    if not recommender or not recommender._initialized:
        raise HTTPException(
            status_code=503, detail="Recommender system not initialized"
        )

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "recommender": recommender._initialized,
            "auth_enabled": settings.ENABLE_AUTH,
            "rate_limiting": True,
        },
    }


@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get Personalized Music Recommendations",
    description="""
    Get personalized music recommendations based on user's listening history using hybrid collaborative filtering and content-based approaches.
    
    **How it works:**
    1. Analyzes your listening history (track sequence)
    2. Uses hybrid CF+CB approach with configurable weights
    3. Applies optional genre/mood filters
    4. Returns ranked recommendations with confidence scores
    5. Supports A/B testing for optimization
    
    **Features:**
    - Sub-millisecond response times with ANN indexing
    - Intelligent caching for performance
    - Real-time A/B testing integration
    - Comprehensive recommendation metadata
    """,
    response_description="Personalized track recommendations with scores and metadata",
    responses={
        200: {
            "description": "Successful recommendations",
            "content": {
                "application/json": {
                    "example": {
                        "recommendations": ["track_123", "track_456", "track_789"],
                        "scores": [0.95, 0.87, 0.82],
                        "metadata": {
                            "cf_weight": 0.7,
                            "cb_weight": 0.3,
                            "session_length": 5,
                            "fusion_method": "weighted_sum",
                            "cf_available": True,
                            "cb_available": True,
                            "experiment_id": "exp_cf_weight_test",
                            "processing_time_ms": 15.4,
                        },
                    }
                }
            },
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "fusion_method"],
                                "msg": "fusion_method must be one of: weighted_sum, max, rank_fusion",
                                "type": "value_error",
                            }
                        ]
                    }
                }
            },
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Please try again later."
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable",
            "content": {
                "application/json": {
                    "example": {"detail": "Recommender system not initialized"}
                }
            },
        },
    },
)
async def get_recommendations(
    request: RecommendationRequest,
    http_request: Request,
    user: str = Depends(verify_api_key),
):
    global recommender, experimental_recommender
    start_time = time.time()
    cache_hit = False

    # Rate limiting
    await check_rate_limit(http_request)

    if not recommender or not recommender._initialized:
        raise HTTPException(
            status_code=503, detail="Recommender system not initialized"
        )

    # Get monitoring components
    cache_manager = get_cache_manager()
    metrics_collector = get_metrics_collector()

    # Determine user ID for A/B testing (use authenticated user or session-based ID)
    user_id = (
        user
        if user != "anonymous"
        else f"session_{http_request.client.host}_{int(time.time())}"
    )

    # Check cache first
    cache_key = cache_manager.get_recommendation_cache_key(
        track_sequence=request.track_sequence,
        top_k=request.top_k,
        fusion_method=request.fusion_method,
        genre_filter=request.genre_filter,
        mood_filter=request.mood_filter,
    )

    cached_result = cache_manager.get_cached_recommendations(cache_key)

    try:
        if cached_result:
            # Cache hit
            result = cached_result
            cache_hit = True
            logger.debug(f"Cache hit for key: {cache_key}")
        else:
            # Cache miss - generate recommendations using experimental recommender
            if experimental_recommender:
                result = experimental_recommender.get_recommendations(
                    user_id=user_id,
                    track_sequence=request.track_sequence,
                    k=request.top_k,
                    genre_filter=request.genre_filter,
                    mood_filter=request.mood_filter,
                    fusion_method=request.fusion_method,
                )
            else:
                # Fallback to base recommender
                result = recommender.get_recommendations(
                    track_sequence=request.track_sequence,
                    k=request.top_k,
                    genre_filter=request.genre_filter,
                    mood_filter=request.mood_filter,
                    fusion_method=request.fusion_method,
                )

            # Cache the result
            cache_manager.cache_recommendations(cache_key, result)
            logger.debug(f"Cached recommendations for key: {cache_key}")

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000

        # Record request metrics
        request_metrics = RequestMetrics(
            timestamp=time.time(),
            endpoint="/recommendations",
            method="POST",
            status_code=200,
            duration_ms=duration_ms,
            user_id=user,
            session_length=len(request.track_sequence),
            top_k=request.top_k,
            cache_hit=cache_hit,
        )
        metrics_collector.record_request(request_metrics)

        # Record recommendation metrics
        if result.get("scores"):
            rec_metrics = RecommendationMetrics(
                timestamp=time.time(),
                num_recommendations=len(result["recommendations"]),
                avg_score=(
                    sum(result["scores"]) / len(result["scores"])
                    if result["scores"]
                    else 0.0
                ),
                fusion_method=request.fusion_method,
                cf_available=result["metadata"].get("cf_available", False),
                cb_available=result["metadata"].get("cb_available", False),
                session_length=len(request.track_sequence),
                processing_time_ms=duration_ms,
            )
            metrics_collector.record_recommendation(rec_metrics)

        return RecommendationResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in /recommendations: {e}")
        # Record error metrics
        duration_ms = (time.time() - start_time) * 1000
        error_metrics = RequestMetrics(
            timestamp=time.time(),
            endpoint="/recommendations",
            method="POST",
            status_code=400,
            duration_ms=duration_ms,
            user_id=user,
            session_length=len(request.track_sequence),
            top_k=request.top_k,
            cache_hit=cache_hit,
        )
        metrics_collector.record_request(error_metrics)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unhandled error in /recommendations: {e}")
        # Record error metrics
        duration_ms = (time.time() - start_time) * 1000
        error_metrics = RequestMetrics(
            timestamp=time.time(),
            endpoint="/recommendations",
            method="POST",
            status_code=500,
            duration_ms=duration_ms,
            user_id=user,
            session_length=len(request.track_sequence),
            top_k=request.top_k,
            cache_hit=cache_hit,
        )
        metrics_collector.record_request(error_metrics)
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Internal recommendation error")


@app.post("/next_track", response_model=RecommendationResponse)
async def next_track(
    track_sequence: List[str] = Query(...),
    top_k: int = Query(settings.DEFAULT_TOP_K),
    http_request: Request = None,
    user: str = Depends(verify_api_key),
):
    global recommender

    # Rate limiting
    await check_rate_limit(http_request)

    request = RecommendationRequest(track_sequence=track_sequence, top_k=top_k)

    return await get_recommendations(request, http_request, user)


@app.post("/similar_tracks")
async def get_similar_tracks(
    request: SimilarTracksRequest,
    http_request: Request,
    user: str = Depends(verify_api_key),
):
    global recommender

    # Rate limiting
    await check_rate_limit(http_request)

    if not recommender or not recommender._initialized:
        raise HTTPException(
            status_code=503, detail="Recommender system not initialized"
        )

    try:
        result = recommender.get_track_similarities(
            track_id=request.track_id,
            k=request.top_k,
            use_cf=request.use_cf,
            use_cb=request.use_cb,
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error in /similar_tracks: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled error in /similar_tracks: {e}")
        raise HTTPException(status_code=500, detail="Internal similarity error")


@app.get(
    "/stats",
    response_model=SystemStatsResponse,
    tags=["System"],
    summary="Get System Statistics",
    description="""
    Retrieve comprehensive system performance and health statistics.
    
    **Statistics Include:**
    - Model performance metrics
    - System resource usage
    - Request processing stats
    - Component health status
    
    **Use Cases:**
    - System monitoring
    - Performance optimization
    - Health checks
    - Capacity planning
    """,
    response_description="Comprehensive system statistics",
)
async def get_system_stats(user: str = Depends(verify_api_key)):
    global recommender

    if not recommender:
        return SystemStatsResponse(status="not_initialized")

    stats = recommender.get_system_stats()
    return SystemStatsResponse(**stats)


@app.get(
    "/tracks",
    tags=["Data"],
    summary="List Available Tracks",
    description="""
    Retrieve paginated list of available tracks in the system.
    
    **Parameters:**
    - `limit`: Maximum number of tracks to return (default: 100)
    - `offset`: Number of tracks to skip for pagination (default: 0)
    
    **Returns:**
    - Paginated track list
    - Total count
    - Pagination metadata
    
    **Use Cases:**
    - Browse available content
    - Data exploration
    - Integration testing
    - Content management
    """,
    response_description="Paginated list of tracks with metadata",
)
async def list_tracks(
    limit: int = 100, offset: int = 0, user: str = Depends(verify_api_key)
):
    global recommender

    if not recommender or not recommender._initialized:
        raise HTTPException(
            status_code=503, detail="Recommender system not initialized"
        )

    track_ids = recommender.vectorizer.get_track_ids()
    total = len(track_ids)

    # Apply pagination
    paginated_tracks = track_ids[offset : offset + limit]

    return {
        "tracks": paginated_tracks,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }


# Monitoring Endpoints
@app.get(
    "/monitoring/health",
    tags=["Monitoring"],
    summary="Comprehensive Health Check",
    description="""
    Advanced health check that provides detailed information about all system components.
    
    **Includes:**
    - Recommender system status
    - Model initialization status
    - Memory usage and performance
    - Cache system status
    - Database connectivity (if applicable)
    """,
    response_description="Comprehensive system health report",
)
async def detailed_health_check():
    """Comprehensive health check with component details"""
    health_checker = get_health_checker()
    return health_checker.run_health_checks()


@app.get(
    "/monitoring/metrics",
    tags=["Monitoring"],
    summary="Get Metrics Summary",
    description="""
    Retrieve aggregated performance metrics and system statistics.
    
    **Metrics Include:**
    - Request/response latencies
    - Cache hit rates
    - Recommendation accuracy
    - Error rates
    - Throughput statistics
    """,
    response_description="System performance metrics",
)
async def get_metrics_summary():
    """Get metrics summary"""
    metrics_collector = get_metrics_collector()
    return metrics_collector.get_summary_stats()


@app.get(
    "/monitoring/prometheus",
    response_class=PlainTextResponse,
    tags=["Monitoring"],
    summary="Prometheus Metrics Export",
    description="""
    Export metrics in Prometheus format for monitoring and alerting systems.
    
    **Use with:**
    - Prometheus monitoring
    - Grafana dashboards
    - Alert manager
    - Custom monitoring tools
    """,
    response_description="Prometheus-formatted metrics",
)
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics_collector = get_metrics_collector()
    prometheus_data = metrics_collector.get_prometheus_metrics()

    if prometheus_data is None:
        raise HTTPException(status_code=501, detail="Prometheus metrics not available")

    return prometheus_data


@app.get(
    "/monitoring/cache",
    tags=["Monitoring"],
    summary="Get Cache Statistics",
    description="""
    Retrieve detailed cache performance statistics.
    
    **Statistics Include:**
    - Cache hit/miss rates
    - Memory usage
    - Entry counts
    - Eviction statistics
    
    **Use Cases:**
    - Performance optimization
    - Memory usage monitoring
    - Cache efficiency analysis
    """,
    response_description="Cache performance statistics",
)
async def get_cache_stats():
    """Get cache statistics"""
    cache_manager = get_cache_manager()
    return cache_manager.get_cache_stats()


@app.post(
    "/monitoring/cache/clear",
    tags=["Monitoring"],
    summary="Clear Recommendation Cache",
    description="""
    Clear all cached recommendation data to force fresh computation.
    
    **Use Cases:**
    - After model updates
    - Debugging cache issues
    - Performance testing
    - Manual cache maintenance
    
    **Note:** This will temporarily increase response times until cache is rebuilt.
    """,
    response_description="Cache clearing status",
)
async def clear_cache(user: str = Depends(verify_api_key)):
    """Clear the recommendation cache"""
    cache_manager = get_cache_manager()
    success = cache_manager.clear_cache()

    return {
        "success": success,
        "message": "Cache cleared successfully" if success else "Failed to clear cache",
        "timestamp": time.time(),
    }


@app.delete(
    "/monitoring/cache/{pattern}",
    tags=["Monitoring"],
    summary="Invalidate Cache Pattern",
    description="""
    Invalidate cache entries matching a specific pattern.
    
    **Parameters:**
    - `pattern`: Pattern to match cache keys (supports wildcards)
    
    **Examples:**
    - `user_*`: Invalidate all user-specific caches
    - `rec_*`: Invalidate all recommendation caches
    - `similar_*`: Invalidate similarity caches
    
    **Use Cases:**
    - Selective cache invalidation
    - User-specific cache clearing
    - Feature-specific maintenance
    """,
    response_description="Invalidation results",
)
async def invalidate_cache_pattern(pattern: str, user: str = Depends(verify_api_key)):
    """Invalidate cache entries matching pattern"""
    cache_manager = get_cache_manager()
    count = cache_manager.invalidate_pattern(pattern)

    return {"pattern": pattern, "invalidated_count": count, "timestamp": time.time()}


# A/B Testing and Feedback Endpoints
@app.post(
    "/feedback",
    tags=["Feedback"],
    summary="Submit User Feedback",
    description="""
    Submit user feedback for music recommendations to improve the system through learning.
    
    **Feedback Types:**
    - `like`: User liked the recommended track
    - `dislike`: User disliked the track
    - `skip`: User skipped the track without listening
    - `play_complete`: User played the track to completion
    - `explicit_rating`: User provided a rating (1-5 scale via feedback_value)
    
    **Use Cases:**
    - Improving recommendation quality through user preferences
    - A/B testing evaluation and optimization
    - Personalization enhancement
    - Content discovery optimization
    """,
    response_model=FeedbackResponse,
    response_description="Feedback submission confirmation",
    responses={
        200: {
            "description": "Feedback recorded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Feedback recorded successfully",
                        "timestamp": 1692902400.0,
                    }
                }
            },
        },
        422: {
            "description": "Invalid feedback type",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "feedback_type"],
                                "msg": "feedback_type must be one of: like, dislike, skip, play_complete, explicit_rating",
                                "type": "value_error",
                            }
                        ]
                    }
                }
            },
        },
    },
)
async def submit_feedback(
    request: FeedbackRequest, http_request: Request, user: str = Depends(verify_api_key)
):
    """Submit user feedback for recommendations"""
    await check_rate_limit(http_request)

    # Create feedback event
    feedback = FeedbackEvent(
        session_id=request.session_id,
        user_id=request.user_id,
        track_id=request.track_id,
        feedback_type=FeedbackType(request.feedback_type),
        feedback_value=request.feedback_value,
        timestamp=time.time(),
        recommendation_context=request.recommendation_context,
    )

    # Record feedback
    feedback_collector = get_feedback_collector()
    feedback_collector.record_feedback(feedback)

    return {
        "status": "success",
        "message": "Feedback recorded successfully",
        "timestamp": time.time(),
    }


@app.get(
    "/feedback/stats",
    tags=["Feedback"],
    summary="Get Feedback Statistics",
    description="""
    Retrieve aggregated user feedback statistics for analytics and insights.
    
    **Statistics Include:**
    - Total feedback events by type
    - Feedback rates (like rate, skip rate, etc.)
    - Engagement metrics
    - User behavior patterns
    
    **Use Cases:**
    - Monitor user satisfaction
    - Track recommendation quality trends
    - A/B testing evaluation
    - Content strategy optimization
    """,
    response_model=FeedbackStatsResponse,
    response_description="Aggregated feedback statistics",
)
async def get_feedback_stats(user: str = Depends(verify_api_key)):
    """Get feedback statistics"""
    feedback_collector = get_feedback_collector()
    stats = feedback_collector.get_stats()

    return {"feedback_stats": stats, "timestamp": time.time()}


@app.get(
    "/feedback/track/{track_id}",
    tags=["Feedback"],
    summary="Get Track-Specific Feedback",
    description="""
    Retrieve feedback history for a specific track.
    
    **Parameters:**
    - `track_id`: Target track identifier
    - `days`: Number of days to look back (1-365)
    
    **Returns:**
    - Feedback events for the track
    - Aggregated statistics
    - User behavior patterns
    """,
    response_description="Track feedback history and statistics",
)
async def get_track_feedback(
    track_id: str,
    days: int = Query(default=7, ge=1, le=365),
    user: str = Depends(verify_api_key),
):
    """Get feedback for a specific track"""
    feedback_collector = get_feedback_collector()
    feedback_list = feedback_collector.get_track_feedback(track_id, days)

    # Convert to dict format for JSON response
    feedback_data = [
        {
            "session_id": f.session_id,
            "user_id": f.user_id,
            "feedback_type": f.feedback_type.value,
            "feedback_value": f.feedback_value,
            "timestamp": f.timestamp,
        }
        for f in feedback_list
    ]

    return {
        "track_id": track_id,
        "feedback_count": len(feedback_data),
        "feedback_events": feedback_data,
        "days": days,
    }


@app.post(
    "/experiments",
    tags=["Experiments"],
    summary="Create A/B Test Experiment",
    description="""
    Create a new A/B test experiment to optimize recommendation parameters.
    
    **Experiment Features:**
    - **Traffic Splitting**: Control percentage of users in experiment
    - **Configuration Overrides**: Test different parameter values
    - **Success Metrics**: Track specific KPIs (precision, engagement, etc.)
    - **Duration Control**: Set experiment timeline
    
    **Common Use Cases:**
    - Test different CF/CB weight combinations
    - Evaluate new fusion methods
    - Optimize session encoder types
    - Compare ANN vs brute force performance
    
    **Example Config Overrides:**
    ```json
    {
        "cf_weight": 0.8,
        "cb_weight": 0.2,
        "fusion_method": "rank_fusion",
        "session_encoder_type": "lstm"
    }
    ```
    """,
    response_model=ExperimentResponse,
    response_description="Experiment creation confirmation with ID",
    responses={
        200: {
            "description": "Experiment created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "experiment_id": "exp_1692902400",
                        "status": "created",
                        "message": "Experiment created successfully",
                    }
                }
            },
        }
    },
)
async def create_experiment(
    request: ExperimentRequest, user: str = Depends(verify_api_key)
):
    """Create a new A/B experiment"""
    ab_tester = get_ab_tester()

    experiment_id = ab_tester.create_experiment(
        name=request.name,
        description=request.description,
        traffic_split=request.traffic_split,
        config_overrides=request.config_overrides,
        success_metrics=request.success_metrics,
        duration_days=request.duration_days,
    )

    return {
        "experiment_id": experiment_id,
        "status": "created",
        "message": "Experiment created successfully",
    }


@app.post(
    "/experiments/{experiment_id}/start",
    tags=["A/B Testing"],
    summary="Start A/B Test Experiment",
    description="""
    Activate a created A/B test experiment to begin traffic splitting.
    
    **Parameters:**
    - `experiment_id`: Target experiment identifier
    
    **Actions:**
    - Begins routing traffic to experiment variants
    - Starts data collection
    - Activates metric tracking
    
    **Use Cases:**
    - Launch approved experiments
    - Begin data collection phase
    - Start performance comparison
    """,
    response_description="Experiment activation status",
)
async def start_experiment(experiment_id: str, user: str = Depends(verify_api_key)):
    """Start an A/B experiment"""
    ab_tester = get_ab_tester()
    success = ab_tester.start_experiment(experiment_id)

    if success:
        return {
            "experiment_id": experiment_id,
            "status": "started",
            "message": "Experiment started successfully",
        }
    else:
        raise HTTPException(status_code=404, detail="Experiment not found")


@app.post(
    "/experiments/{experiment_id}/stop",
    tags=["A/B Testing"],
    summary="Stop A/B Test Experiment",
    description="""
    Deactivate a running A/B test experiment.
    
    **Parameters:**
    - `experiment_id`: Target experiment identifier
    
    **Actions:**
    - Stops traffic routing to variants
    - Preserves collected data
    - Maintains results for analysis
    
    **Use Cases:**
    - End successful experiments
    - Stop underperforming tests
    - Conclude data collection phase
    """,
    response_description="Experiment deactivation status",
)
async def stop_experiment(experiment_id: str, user: str = Depends(verify_api_key)):
    """Stop an A/B experiment"""
    ab_tester = get_ab_tester()
    success = ab_tester.stop_experiment(experiment_id)

    if success:
        return {
            "experiment_id": experiment_id,
            "status": "stopped",
            "message": "Experiment stopped successfully",
        }
    else:
        raise HTTPException(status_code=404, detail="Experiment not found")


@app.get(
    "/experiments",
    tags=["A/B Testing"],
    summary="List Active Experiments",
    description="""
    Retrieve list of currently active A/B testing experiments.
    
    **Returns:**
    - Experiment configurations
    - Participation statistics
    - Performance metrics
    - Status information
    
    **Use Cases:**
    - Monitor experiment status
    - Check A/B test configurations
    - Analyze experiment performance
    - Debug experiment assignments
    """,
    response_model=ExperimentsListResponse,
    response_description="List of active A/B testing experiments",
)
async def list_experiments(user: str = Depends(verify_api_key)):
    """List all A/B experiments"""
    ab_tester = get_ab_tester()

    experiments = []
    for exp in ab_tester.experiments.values():
        experiments.append(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "description": exp.description,
                "status": exp.status.value,
                "traffic_split": exp.traffic_split,
                "start_date": exp.start_date.isoformat(),
                "end_date": exp.end_date.isoformat() if exp.end_date else None,
                "success_metrics": exp.success_metrics,
            }
        )

    return {
        "experiments": experiments,
        "total_count": len(experiments),
        "timestamp": time.time(),
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
