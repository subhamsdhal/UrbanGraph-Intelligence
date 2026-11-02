"""Tests for Kafka streaming adapter and consumer (R20)."""

import json
import time
import pytest

from urbangraph.data import DataError
from urbangraph.serving.kafka_adapter import (
    InMemoryKafkaBroker,
    KafkaConsumerConfig,
    KafkaEvent,
    KafkaStreamingConsumer,
    KafkaStreamingProducer,
)


def test_kafka_event_serialization() -> None:
    payload = {"region_id": "SYN_A", "load_MW": 142.5}
    event = KafkaEvent(
        topic="urbangraph.telemetry",
        key="SYN_A",
        value=json.dumps(payload).encode("utf-8"),
        timestamp_ms=1700000000000,
        partition=0,
        offset=42,
    )
    assert event.value_as_json() == payload
    assert event.to_dict()["topic"] == "urbangraph.telemetry"
    assert event.offset == 42


def test_kafka_produce_and_consume_streaming() -> None:
    broker = InMemoryKafkaBroker()
    producer = KafkaStreamingProducer(broker=broker)
    consumer = KafkaStreamingConsumer(
        config=KafkaConsumerConfig(group_id="test-group", max_poll_records=10),
        broker=broker,
    )
    consumer.subscribe(["telemetry.grid"])

    # 1. Produce 3 events
    now_ms = int(time.time() * 1000)
    producer.produce("telemetry.grid", {"step": 1}, key="k1", timestamp_ms=now_ms)
    producer.produce("telemetry.grid", {"step": 2}, key="k2", timestamp_ms=now_ms + 100)
    producer.produce("telemetry.grid", {"step": 3}, key="k3", timestamp_ms=now_ms + 200)

    # 2. Poll consumer
    batch = consumer.poll(timeout_ms=100)
    assert len(batch) == 3
    assert batch[0].key == "k1"
    assert batch[0].value_as_json() == {"step": 1}
    assert batch[2].value_as_json() == {"step": 3}

    # 3. Commit offsets and verify second poll returns empty
    consumer.commit()
    batch_empty = consumer.poll(timeout_ms=50)
    assert len(batch_empty) == 0

    # 4. Seek back to offset 1
    consumer.seek("telemetry.grid", 0, 1)
    replayed = consumer.poll(timeout_ms=50)
    assert len(replayed) == 2
    assert replayed[0].offset == 1


def test_kafka_duplicate_and_late_event_rejection() -> None:
    broker = InMemoryKafkaBroker()
    producer = KafkaStreamingProducer(broker=broker)
    consumer = KafkaStreamingConsumer(
        config=KafkaConsumerConfig(
            group_id="watermark-group",
            max_allowed_lateness_ms=5000,  # 5 second watermark window
            deduplicate_keys=True,
        ),
        broker=broker,
    )
    consumer.subscribe(["telemetry.events"])

    t0 = 10000000

    # Produce normal event
    producer.produce("telemetry.events", {"seq": 1}, key="event-1", timestamp_ms=t0)
    # Produce event advancing watermark by 10s
    producer.produce("telemetry.events", {"seq": 2}, key="event-2", timestamp_ms=t0 + 10000)
    # Produce duplicate key event
    producer.produce("telemetry.events", {"seq": 3}, key="event-1", timestamp_ms=t0 + 10500)
    # Produce late event older than watermark window (lag = 10s > 5s limit)
    producer.produce("telemetry.events", {"seq": 4}, key="event-late", timestamp_ms=t0 + 1000)

    batch = consumer.poll(timeout_ms=100)
    # Only seq 1 and seq 2 should be accepted; duplicate seq 3 and late seq 4 dropped
    assert len(batch) == 2
    metrics = consumer.get_metrics()
    assert metrics["records_polled"] == 4
    assert metrics["records_accepted"] == 2
    assert metrics["duplicates_dropped"] == 1
    assert metrics["late_events_dropped"] == 1


def test_kafka_consumer_invalid_subscribe() -> None:
    consumer = KafkaStreamingConsumer(config=KafkaConsumerConfig())
    with pytest.raises(DataError, match="empty topic list"):
        consumer.subscribe([])

    with pytest.raises(DataError, match="non-negative"):
        consumer.seek("topic", 0, -1)
