"""Optional error tracking with Sentry. Off unless SENTRY_DSN is set.

A gateway sees prompts, and prompts carry personal data, so the reports are kept to the
error itself: no request bodies, no headers or cookies (default PII off), no local
variables from the stack, and no performance tracing.
"""

import logging

from gateway.config import Settings

log = logging.getLogger("gateway")


def init_error_tracking(settings: Settings) -> bool:
    if not settings.sentry_dsn:
        return False
    import sentry_sdk  # imported only when it's used

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        send_default_pii=False,
        max_request_body_size="never",
        include_local_variables=False,
        traces_sample_rate=0.0,
    )
    log.info('{"event": "error_tracking_enabled", "provider": "sentry"}')
    return True
