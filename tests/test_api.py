def test_health_endpoint_importable():
    """Smoke test: import the API module and ensure `app` is present.

    If FastAPI is installed this will be a FastAPI app; otherwise the module
    exposes a WSGI fallback `app` callable. This test ensures the entrypoint
    is importable in CI without requiring the full runtime server.
    """
    import importlib
    mod = importlib.import_module('api.index')
    assert hasattr(mod, 'app'), "api.index must export `app`"


def test_health_response():
    """If FastAPI is available, run a functional health check using TestClient."""
    try:
        from fastapi.testclient import TestClient
    except Exception:
        # FastAPI not installed in this environment; skip functional check.
        return

    import api.index as index
    client = TestClient(index.app)
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json().get('status') == 'ok'
