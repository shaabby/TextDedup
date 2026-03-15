from textdedup.cpp_bridge import CppBridge, SimHashFingerprint, WeightedDocument


def test_compute_simhash_is_deterministic_in_python_fallback() -> None:
    bridge = CppBridge(prefer_cpp=False)
    docs = [
        WeightedDocument(doc_id="a", tokens=["北京", "天气", "晴朗"], weights=[1.0, 2.0, 1.5]),
        WeightedDocument(doc_id="b", tokens=["北京", "阳光", "明媚"], weights=[1.0, 1.5, 2.0]),
    ]

    run1 = bridge.compute_simhash(docs)
    run2 = bridge.compute_simhash(docs)

    assert [x.fingerprint for x in run1] == [x.fingerprint for x in run2]


def test_hamming_topk_returns_sorted_candidates() -> None:
    bridge = CppBridge(prefer_cpp=False)
    query = [SimHashFingerprint(doc_id="q", hash_bits=64, fingerprint=0b1111)]
    corpus = [
        SimHashFingerprint(doc_id="a", hash_bits=64, fingerprint=0b1111),
        SimHashFingerprint(doc_id="b", hash_bits=64, fingerprint=0b0111),
        SimHashFingerprint(doc_id="c", hash_bits=64, fingerprint=0b0000),
    ]

    results = bridge.hamming_topk(query, corpus, top_k=3, max_distance=64, exclude_self=False)
    assert len(results) == 1
    assert [c.doc_id for c in results[0].candidates] == ["a", "b", "c"]
    assert results[0].meta.engine == "python"


def test_bridge_marks_fallback_when_cpp_unavailable() -> None:
    bridge = CppBridge(prefer_cpp=True)
    query = [SimHashFingerprint(doc_id="q", hash_bits=64, fingerprint=0b1010)]
    corpus = [SimHashFingerprint(doc_id="d", hash_bits=64, fingerprint=0b1000)]

    result = bridge.hamming_topk(query, corpus, top_k=1)
    assert result[0].meta.fallback_used is True


def test_compute_simhash_rejects_mismatched_lengths() -> None:
    bridge = CppBridge(prefer_cpp=False)
    bad = [WeightedDocument(doc_id="x", tokens=["a", "b"], weights=[1.0])]

    try:
        bridge.compute_simhash(bad)
        assert False, "expected ValueError"
    except ValueError as ex:
        assert "长度" in str(ex)
