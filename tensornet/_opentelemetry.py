# Copyright 2020-2025 Qihoo Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from typing import Iterator
from opentelemetry import context
from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.propagators.textmap import DefaultGetter
from opentelemetry.sdk._configuration import _OTelSDKConfigurator
from opentelemetry.sdk.resources import Resource, get_aggregated_resources, ProcessResourceDetector
from opentelemetry.util.types import Attributes

from . import version as _version

_RESK_TN_VER = "tensornet.version"
logger = logging.getLogger(__name__)

if os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    # initialize opentelemetry
    os.environ["OTEL_TRACES_EXPORTER"] = "otlp"
    os.environ["OTEL_SERVICE_NAME"] = "tensornet"
    try:
        _OTelSDKConfigurator().configure(
            auto_instrumentation_version="tensornet-otel-auto-config",  # set telemetry.auto.version
            resource_attributes={_RESK_TN_VER, (_version.VERSION)},
        )
        provider = trace.get_tracer_provider()
        append_resource = get_aggregated_resources(
            [ProcessResourceDetector()], Resource({_RESK_TN_VER: _version.VERSION})
        )
        provider._resource = provider.resource.merge(append_resource)
    except RuntimeError:
        logger.warning("No opentelemetry trace exporter implementations")


def start_as_current_span(name: str, attributes: Attributes = None) -> Iterator[trace.Span]:
    tracer = trace.get_tracer("tensornet", _version.VERSION)  # set otel.library.name and otel.library.version

    class EnvGetter(DefaultGetter):
        def get(self, carrier, key):
            return super().get(carrier, key.upper())

    ctx = extract(os.environ, context.get_current(), EnvGetter())

    # operation = name
    return tracer.start_as_current_span(name, context=ctx, attributes=attributes)
