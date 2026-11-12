# Copyright 2010 New Relic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import weakref

from newrelic.api.application import application_instance
from newrelic.api.background_task import BackgroundTask


class Stream:
    pass


def test_registered_stream_does_not_keep_transaction_alive():
    freed = []
    gc.collect()
    gc.disable()
    try:
        transaction = BackgroundTask(application_instance(), "test_registered_stream_does_not_keep_transaction_alive")
        stream = Stream()
        transaction._register_llm_stream_finalizer(stream, lambda s, t: None)
        weakref.ref(transaction, lambda _: freed.append(True))
        del transaction
        assert freed, "Transaction was not freed without a garbage collection pass"
    finally:
        gc.enable()


def test_finalizer_receives_stream_and_transaction_at_exit():
    calls = []
    stream = Stream()
    with BackgroundTask(application_instance(), "test_finalizer_receives_stream_and_transaction_at_exit") as txn:
        txn._register_llm_stream_finalizer(stream, lambda s, t: calls.append((s, t)))
    assert calls == [(stream, txn)]


def test_failing_finalizer_is_logged_and_does_not_stop_others(caplog):
    calls = []
    streams = [Stream(), Stream()]

    def boom(stream, transaction):
        raise RuntimeError("finalizer failure")

    with caplog.at_level(logging.DEBUG, logger="newrelic.api.transaction"):
        with BackgroundTask(application_instance(), "test_failing_finalizer_is_logged") as txn:
            txn._register_llm_stream_finalizer(streams[0], lambda s, t: calls.append("ok"))
            txn._register_llm_stream_finalizer(streams[1], boom)

    assert calls == ["ok"]
    assert any(r.exc_info and "finalizer failure" in str(r.exc_info[1]) for r in caplog.records)
