import zstandard

from analysis.real_dashboard.tail import LogTail


def test_a_plain_log_is_read_incrementally_and_a_partial_line_waits(*, tmp_path):
    path = tmp_path / "step_events.jsonl"
    tail = LogTail(path=path)
    assert tail.read().lines == []
    path.write_text('{"a": 1}\n{"b": 2')
    first = tail.read()
    assert first.lines == ['{"a": 1}']
    assert not first.restarted
    with path.open("a") as stream:
        stream.write('}\n{"c": 3}\n')
    assert tail.read().lines == ['{"b": 2}', '{"c": 3}']
    assert tail.read().lines == []


def test_a_truncated_log_is_read_again_from_the_start(*, tmp_path):
    path = tmp_path / "robot_session.log"
    path.write_text("one\ntwo\nthree\n")
    tail = LogTail(path=path)
    assert tail.read().lines == ["one", "two", "three"]
    path.write_text("new\n")
    again = tail.read()
    assert again.restarted
    assert again.lines == ["new"]


def test_a_zst_log_written_one_frame_per_record_is_read_frame_by_frame(*, tmp_path):
    compressor = zstandard.ZstdCompressor(level=3)
    path = tmp_path / "pomdp_decisions.jsonl.zst"
    path.write_bytes(compressor.compress(b'{"event": "decision"}\n'))
    # The plain name resolves to the `.zst` sibling, as `JsonlLog.resolve` does.
    tail = LogTail(path=tmp_path / "pomdp_decisions.jsonl")
    assert tail.read().lines == ['{"event": "decision"}']
    frame = compressor.compress(b'{"event": "outcome"}\n')
    with path.open("ab") as stream:
        stream.write(frame[:5])
    assert tail.read().lines == []
    with path.open("ab") as stream:
        stream.write(frame[5:])
    assert tail.read().lines == ['{"event": "outcome"}']
