"""Tests for agent.thread_scoped_output.thread_scoped_silence.

Behaviour contract: a thread inside ``thread_scoped_silence()`` has its
stdout/stderr routed to devnull, while every OTHER thread keeps writing to the
real stream — even concurrently, while the first thread is still inside the
context.  This is the property the old process-global
``contextlib.redirect_stdout(devnull)`` violated (issue #55769 / #55925).
"""

import io
import sys
import threading
import time

import pytest

from agent.thread_scoped_output import thread_scoped_silence


def _run_with_real_stream(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()


def test_current_thread_is_silenced():
    def body():
        with thread_scoped_silence():
            print("dropped")
        print("kept")

    captured = _run_with_real_stream(body)
    assert "dropped" not in captured
    assert "kept" in captured


def test_concurrent_thread_keeps_output_during_silence_window():
    """A loud thread writing WHILE another thread is silenced must survive."""
    inside_silence = threading.Event()
    loud_done = threading.Event()

    def silenced_worker():
        with thread_scoped_silence():
            print("SILENCED")
            inside_silence.set()
            # Hold the silence window until the loud thread has written.
            loud_done.wait(timeout=2.0)

    def loud_worker():
        inside_silence.wait(timeout=2.0)
        print("LOUD")
        loud_done.set()

    def body():
        t1 = threading.Thread(target=silenced_worker)
        t2 = threading.Thread(target=loud_worker)
        t1.start()
        t2.start()
        t1.join(timeout=15.0)
        t2.join(timeout=15.0)
        assert not t1.is_alive() and not t2.is_alive(), "worker threads didn't finish"

    captured = _run_with_real_stream(body)
    assert "SILENCED" not in captured
    assert "LOUD" in captured


def test_stderr_is_also_routed_per_thread():
    real_err = io.StringIO()
    orig = sys.stderr
    sys.stderr = real_err
    try:
        with thread_scoped_silence():
            sys.stderr.write("err-dropped\n")
        sys.stderr.write("err-kept\n")
    finally:
        sys.stderr = orig
    out = real_err.getvalue()
    assert "err-dropped" not in out
    assert "err-kept" in out


def test_nested_silence_same_thread_composes():
    def body():
        with thread_scoped_silence():
            with thread_scoped_silence():
                print("inner")
            # Still inside the OUTER context — depth-counted, so this thread
            # remains silenced after the inner context exits.
            print("after-inner")
        print("after-outer")

    captured = _run_with_real_stream(body)
    assert "inner" not in captured
    assert "after-inner" not in captured
    assert "after-outer" in captured


def test_unsilence_cleans_up_after_exit():
    """After the context exits, the calling thread writes to the real stream."""
    seen = []

    def body():
        with thread_scoped_silence():
            pass
        print("post")
        seen.append("post")

    captured = _run_with_real_stream(body)
    assert "post" in captured
    assert seen == ["post"]


def test_many_concurrent_silenced_and_loud_threads():
    """Stress: interleaved silenced/loud threads keep their respective fates."""
    start = threading.Event()
    results_lock = threading.Lock()

    def silenced(i):
        start.wait(timeout=2.0)
        with thread_scoped_silence():
            print(f"S{i}")
            time.sleep(0.05)

    def loud(i):
        start.wait(timeout=2.0)
        time.sleep(0.02)
        print(f"L{i}")

    def body():
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=silenced, args=(i,)))
            threads.append(threading.Thread(target=loud, args=(i,)))
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=15.0)
        assert not any(t.is_alive() for t in threads), "straggler thread would truncate captured output"

    captured = _run_with_real_stream(body)
    for i in range(5):
        assert f"S{i}" not in captured, f"silenced S{i} leaked"
        assert f"L{i}" in captured, f"loud L{i} swallowed"


# ---------------------------------------------------------------------------
# Sink lifetime
# ---------------------------------------------------------------------------
#
# The routing proxy is installed once and deliberately never uninstalled, so it
# stays bound to sys.stdout/sys.stderr after a block exits. The devnull sink it
# routes silenced threads to therefore has to outlive every block. When each
# block owned (and closed) the sink, the first block left the still-installed
# proxy pointing at a CLOSED file: from the second block onward every silenced
# thread wrote into it. write()/flush() swallow the resulting ValueError, but
# fileno() does not — and subprocess spawning with ``stdout=sys.stdout`` (as
# tools/mcp_stdio_watchdog.py does) calls exactly that. background_review.py
# opens two such blocks, so a real run hits this on its second silencing.


@pytest.fixture()
def installed_registry():
    """Isolate the module's install state.

    Reads the registry out of ``thread_scoped_silence.__globals__`` rather than
    re-importing: another test in the suite reloads this module, which would
    otherwise hand the fixture a *different* module object than the function
    under test writes into. ``_ensure_installed`` also reinstalls whenever
    ``sys.stdout`` is no longer the proxy it put there, so pytest's capture
    would leak across these assertions without the rebind below.
    """
    # ``@contextlib.contextmanager`` wraps the function, so reach the original
    # via __wrapped__ to land in the defining module's namespace.
    _fn = getattr(thread_scoped_silence, "__wrapped__", thread_scoped_silence)
    registry = _fn.__globals__["_installed"]

    saved_out, saved_err = sys.stdout, sys.stderr
    saved_installed = dict(registry)
    registry.clear()
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield registry
    finally:
        registry.clear()
        registry.update(saved_installed)
        sys.stdout, sys.stderr = saved_out, saved_err


def test_sink_survives_the_first_block(installed_registry):
    with thread_scoped_silence():
        pass

    proxy = installed_registry["stdout"]

    assert not proxy._sink.closed, (
        "the installed proxy outlives the block, so its sink must too"
    )


def test_second_silenced_block_can_still_use_fileno(installed_registry):
    """subprocess(stdout=sys.stdout) inside a silenced thread must not blow up."""
    with thread_scoped_silence():
        pass

    captured = {}

    def worker():
        try:
            with thread_scoped_silence():
                captured["fd"] = sys.stdout.fileno()
        except Exception as exc:  # noqa: BLE001 — recorded for the assertion
            captured["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "error" not in captured, captured.get("error")
    assert isinstance(captured.get("fd"), int)


def test_repeated_blocks_reuse_one_sink(installed_registry):
    """The sink is process-scoped; a per-call sink was also a per-call fd."""
    with thread_scoped_silence():
        pass

    first = installed_registry["stdout"]._sink

    for _ in range(5):
        with thread_scoped_silence():
            pass

    assert installed_registry["stdout"]._sink is first
    assert not first.closed


def test_silencing_still_works_after_many_blocks():
    """The lifetime fix must not weaken the actual silencing contract."""
    for _ in range(3):
        with thread_scoped_silence():
            pass

    def worker():
        with thread_scoped_silence():
            print("SILENCED-LATE-BLOCK")

    reached = _run_with_real_stream(worker)

    assert "SILENCED-LATE-BLOCK" not in reached
