import pytest
from routers.health import liveness

@pytest.mark.asyncio
async def test_liveness_probe():
    assert await liveness() == {"status": "alive"}
