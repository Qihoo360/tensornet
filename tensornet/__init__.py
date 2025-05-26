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

from . import core as core
from . import feature_column as feature_column
from . import layers as layers
from . import distribute as distribute
from . import callbacks as callbacks
from . import metric as metric
from . import optimizer as optimizer
from . import model as model
from . import data as data

from . import version as _version

__version__ = _version.VERSION
