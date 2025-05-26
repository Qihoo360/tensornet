# -*- coding: utf-8 -*-

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

import tensornet as tn


from tensorflow.python.distribute.one_device_strategy import OneDeviceStrategy, OneDeviceExtended


class PsStrategy(OneDeviceStrategy):
    """ """

    def __init__(self):
        super(OneDeviceStrategy, self).__init__(PsExtend(self))


class PsExtend(OneDeviceExtended):
    """ """

    def __init__(self, container_strategy):
        tn.core.init()

        super(PsExtend, self).__init__(container_strategy, "/cpu:0")

        return

    @property
    def should_checkpoint(self):
        return tn.core.self_shard_id() == 0

    def _validate_colocate_with_variable(self, colocate_with_variable):
        """ """
        pass

    def _experimental_assign_to_logical_device(self, tensor, logical_device_id):
        raise NotImplementedError("not implement")

    def _experimental_split_to_logical_devices(self, tensor, partition_dimensions):
        raise NotImplementedError("not implement")

    def _experimental_replicate_to_logical_devices(self, tensor):
        raise NotImplementedError("not implement")

    def variable_created_in_scope(self, v):
        return True
