"""Jev results must survive the Hermes automatic context injection filter."""
import pytest

from hermes_memory_provider import MnemosyneMemoryProvider


@pytest.mark.parametrize('source,prefix', [('conversation', '[USER] '), ('fact', '')])
def test_automatic_prefetch_uses_jev_relevance_without_embeddings(source, prefix):
    class Beam:
        author_id = None

        def recall(self, **kwargs):
            assert kwargs['query'] == 'Which database does Cedar use?'
            return [dict(content=prefix + 'Cedar uses PostgreSQL.', source=source,
                         score=.94, jev_relevance=.94, dense_score=0., fts_score=0.,
                         importance=.5),
                    dict(content='An unrelated important memory.', source='fact',
                         score=.95, importance=1., dense_score=0., fts_score=0.)]

    provider = MnemosyneMemoryProvider()
    provider._beam = Beam()
    block = provider.prefetch('Which database does Cedar use?')
    assert '## PerfectRecall Context' in block
    assert 'Cedar uses PostgreSQL.' in block
    assert 'unrelated' not in block
