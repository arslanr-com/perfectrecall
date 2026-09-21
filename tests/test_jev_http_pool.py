"""Pooling must retain endpoint, retry, redirect and response-size protections."""
import json

import httpx
import pytest

from mnemosyne.core import jev


def pooled_client(monkeypatch, handler):
    original = httpx.Client
    pools = []
    def create(**kwargs):
        pool = original(transport=httpx.MockTransport(handler), **kwargs)
        pools.append(pool)
        return pool
    monkeypatch.setattr(httpx, 'Client', create)
    monkeypatch.setenv('MNEMOSYNE_JEV_HTTP_TRANSPORT', 'pooled')
    return jev.JevClient('test-key'), pools


def test_requests_reuse_client_and_preserve_typed_payload(monkeypatch):
    seen = []
    def handler(request):
        assert str(request.url) == 'https://openrouter.ai/api/alpha/decisions'
        assert request.headers['authorization'] == 'Bearer test-key'
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=dict(model='test', answers={key: dict(type='noul', noul=.9)
                              for key in payload['questions']}, usage={}))
    client, pools = pooled_client(monkeypatch, handler)
    for state in ['first', 'second']:
        assert client.evaluate(state, {'0': jev.noul('Relevant?')})['0']['noul'] == .9
    assert len(pools) == 1
    assert [body['state'] for body in seen] == ['first', 'second']
    client.close()
    assert pools[0].is_closed


def test_redirect_never_receives_credentials(monkeypatch):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(307, headers={'Location': 'https://other.invalid/collect'})
    client, _ = pooled_client(monkeypatch, handler)
    with pytest.raises(jev.JevError, match='307'):
        client.evaluate('data', {'0': jev.noul('Relevant?')})
    assert seen == ['https://openrouter.ai/api/alpha/decisions']
    client.close()


def test_retry_after_remains_case_insensitive(monkeypatch):
    attempts, sleeps = [], []
    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={'retry-after': '3'})
        return httpx.Response(200, json=dict(model='test', answers={'0': dict(type='noul', noul=.9)}, usage={}))
    client, _ = pooled_client(monkeypatch, handler)
    monkeypatch.setattr(jev.time, 'sleep', sleeps.append)
    client.evaluate('data', {'0': jev.noul('Relevant?')})
    assert sleeps == [3] and len(attempts) == 2
    client.close()


def test_oversized_response_fails_closed(monkeypatch):
    client, _ = pooled_client(monkeypatch, lambda request: httpx.Response(200, content=b' ' * 4_000_001))
    with pytest.raises(jev.JevError, match='size limit'):
        client.evaluate('data', {'0': jev.noul('Relevant?')})
    assert client.snapshot()['failures'] == 1
    client.close()
