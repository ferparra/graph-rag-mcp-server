"""
Resilience utilities for MCP server - retry logic, circuit breakers, and error recovery.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, Union, Dict, Awaitable, cast
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise e
    
    async def async_call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback when retry occurs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {e}"
                        )
            
            if last_exception:
                raise last_exception
            raise Exception(f"Unexpected retry failure for {func.__name__}")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {e}"
                        )
            
            if last_exception:
                raise last_exception
            raise Exception(f"Unexpected retry failure for {func.__name__}")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ConnectionPool:
    """Simple connection pool for managing database connections."""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 2
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self._pool: list = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the pool with minimum connections."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.min_size):
                conn = await self._create_connection()
                self._pool.append(conn)
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {self.min_size} connections")
    
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            # Try to get an existing connection
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(id(conn))
                return conn
            
            # Create new connection if under limit
            if len(self._in_use) < self.max_size:
                conn = await self._create_connection()
                self._in_use.add(id(conn))
                return conn
        
        # Wait and retry if at capacity
        await asyncio.sleep(0.1)
        return await self.acquire()
    
    async def release(self, conn: Any):
        """Release a connection back to the pool."""
        async with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                
                # Return to pool if healthy, otherwise discard
                if await self._is_healthy(conn):
                    self._pool.append(conn)
                else:
                    logger.warning("Discarding unhealthy connection")
                    try:
                        await self._close_connection(conn)
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
    
    async def _create_connection(self) -> Any:
        """Create a new connection."""
        if asyncio.iscoroutinefunction(self.factory):
            return await self.factory()
        return self.factory()
    
    async def _is_healthy(self, conn: Any) -> bool:
        """Check if a connection is healthy."""
        # Override in subclasses for specific health checks
        return conn is not None
    
    async def _close_connection(self, conn: Any):
        """Close a connection."""
        if hasattr(conn, 'close'):
            if asyncio.iscoroutinefunction(conn.close):
                await conn.close()
            else:
                conn.close()
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool:
                try:
                    await self._close_connection(conn)
                except Exception as e:
                    logger.error(f"Error closing pooled connection: {e}")
            self._pool.clear()
            self._in_use.clear()
            self._initialized = False


class RateLimiter:
    """Token bucket rate limiter for request throttling."""
    
    def __init__(
        self,
        rate: float,  # Requests per second
        burst: int = 1  # Maximum burst size
    ):
        self.rate = rate
        self.burst = burst
        self.tokens: float = float(burst)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens for a request.
        Returns True if successful, False if rate limited.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(
                float(self.burst),
                self.tokens + elapsed * self.rate
            )
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_and_acquire(self, tokens: int = 1):
        """Wait until tokens are available and acquire them."""
        while not await self.acquire(tokens):
            wait_time = tokens / self.rate
            await asyncio.sleep(wait_time)


def timeout_decorator(seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        return wrapper
    return decorator


class HealthCheck:
    """Health check manager for monitoring service health."""
    
    def __init__(self):
        self.checks: Dict[str, Any] = {}  # Using Any to handle both sync and async functions
        self.last_results: Dict[str, tuple[bool, str, float]] = {}
    
    def register(
        self,
        name: str,
        check_func: Union[Callable[[], Union[bool, tuple[bool, str]]], Callable[[], Awaitable[Union[bool, tuple[bool, str]]]]]
    ):
        """Register a health check function."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            start_time = time.time()
            try:
                # Handle both sync and async check functions
                if asyncio.iscoroutinefunction(check_func):
                    # Type assertion: we know this is an async function
                    async_func = cast(Callable[[], Awaitable[Union[bool, tuple[bool, str]]]], check_func)
                    result = await async_func()
                else:
                    # Type assertion: we know this is a sync function
                    sync_func = cast(Callable[[], Union[bool, tuple[bool, str]]], check_func)
                    result = sync_func()
                
                if isinstance(result, tuple):
                    healthy, message = result
                else:
                    healthy = bool(result)
                    message = "OK" if healthy else "Failed"
                
                duration = time.time() - start_time
                self.last_results[name] = (bool(healthy), str(message), duration)
                
                results[name] = {
                    "healthy": healthy,
                    "message": message,
                    "duration_ms": round(duration * 1000, 2)
                }
                
                if not healthy:
                    overall_healthy = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.last_results[name] = (False, str(e), duration)
                results[name] = {
                    "healthy": False,
                    "message": f"Check failed: {e}",
                    "duration_ms": round(duration * 1000, 2)
                }
                overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": time.time()
        }


# Graceful degradation utilities
class GracefulDegradation:
    """Utilities for graceful degradation when services are unavailable."""
    
    @staticmethod
    def with_fallback(
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T],
        log_fallback: bool = True
    ) -> Callable:
        """Execute primary function with fallback on failure."""
        
        @functools.wraps(primary_func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"Primary function {primary_func.__name__} failed: {e}. "
                        f"Using fallback {fallback_func.__name__}"
                    )
                return fallback_func(*args, **kwargs)
        
        @functools.wraps(primary_func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    return await primary_func(*args, **kwargs)
                else:
                    return primary_func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"Primary function {primary_func.__name__} failed: {e}. "
                        f"Using fallback {fallback_func.__name__}"
                    )
                
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(primary_func):
            return async_wrapper
        return wrapper
    
    @staticmethod
    def cache_on_failure(
        cache_key_func: Callable[..., str],
        cache_ttl: float = 3600.0
    ):
        """Decorator to return cached results when primary function fails."""
        cache: Dict[str, tuple[Any, float]] = {}
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                cache_key = cache_key_func(*args, **kwargs)
                
                try:
                    result = func(*args, **kwargs)
                    # Update cache with fresh result
                    cache[cache_key] = (result, time.time())
                    return result
                except Exception as e:
                    # Try to return cached result
                    if cache_key in cache:
                        cached_result, cached_time = cache[cache_key]
                        if time.time() - cached_time <= cache_ttl:
                            logger.warning(
                                f"{func.__name__} failed: {e}. "
                                f"Returning cached result for key {cache_key}"
                            )
                            return cached_result
                    raise e
            
            return wrapper
        return decorator