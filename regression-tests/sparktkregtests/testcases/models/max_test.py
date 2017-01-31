# vim: set encoding=utf-8

#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Tests accuracy of max train and predict methods"""

import unittest
from sparktkregtests.lib import sparktk_test
import os
import sys

class MaxTest(sparktk_test.SparkTKTestCase):
    def setUp(self):
        super(MaxTest, self).setUp()
        schema = [("Int", int),
                  ("output", float),
                  ("exo1", float),
                  ("exo2", float),
                  ("exo3", float)]
        dataset = self.get_file("max_data.csv")
        self.frame = self.context.frame.import_csv(dataset, schema=schema, delimiter=",", header=False)


    def test_max_train(self):
        """Test MAx train method"""
        train_frame = self.frame.copy(where= lambda row: row.Int <= 499990)
        ts_column = "output"
        x_columns = ["exo1", "exo2", "exo3"]
        model = self.context.models.timeseries.max.train(train_frame, ts_column, x_columns, 1, 0)
        print model
        print "8.0, 0.566, 0.12, 0.65, -0.333"

        #These numbers are copied from the max_gen.py file in generatedata
        self.assertAlmostEqual(model.c, 8.0, delta=0.001)
        self.assertAlmostEqual(model.ma[0], 0.566, delta=0.001)
        self.assertAlmostEqual(model.xreg[0], 0.12, delta=0.001)
        self.assertAlmostEqual(model.xreg[1], -0.65, delta=0.001)
        self.assertAlmostEqual(model.xreg[2], -0.333, delta=0.001)

if __name__ == "__main__":
    unittest.main()
