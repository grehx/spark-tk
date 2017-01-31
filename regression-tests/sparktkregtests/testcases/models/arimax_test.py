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

"""Tests accuracy of arimax train and predict"""

import unittest
from sparktkregtests.lib import sparktk_test
import os
import sys


class ArimaxTest(sparktk_test.SparkTKTestCase):
    def setUp(self):
        super(ArimaxTest, self).setUp()
        schema = [("Int", int),
                  ("output1", float),
                  ("output2", float),
                  ("exo1", float),
                  ("exo2", float),
                  ("exo3", float)]
        dataset = self.get_file("arimax_data.csv")
        self.frame = self.context.frame.import_csv(dataset, schema=schema, delimiter=",", header=False)

    def test_arimax_train(self):
        """Test ARIMAx train method"""
        self.assertEqual(500000, self.frame.count(), msg= "Dataframe was not created correctly. Check dataset does not include infinite")
        train_frame = self.frame.copy(where= lambda row: row.Int <= 499990)
        ts_column ="output1"
        x_columns = ["exo1", "exo2", "exo3"]
        #arimax.train(frame, ts_column, x_columns, p, d, q, x_max_lag, include_original_x=True, include_intercept=True, init_params=None)
        model = self.context.models.timeseries.arimax.train(train_frame, ts_column, x_columns, 1, 1, 1, 0, True, True)

        #These values are copied from the arimax datagen file.
        self.assertAlmostEqual(model.c, 1.335, delta=0.01)
        self.assertAlmostEqual(model.ar[0], 0.542, delta=0.001)
        self.assertAlmostEqual(model.ma[0], 0.266, delta=0.001)
        self.assertAlmostEqual(model.xreg[0], 0.1293, delta=0.001)
        self.assertAlmostEqual(model.xreg[1], 0.0781, delta=0.001)
        self.assertAlmostEqual(model.xreg[2], -0.04275, delta=0.001)

    def test_arima_train(self):
        """Test ARIMA train method"""
        self.assertEqual(500000, self.frame.count(), msg= "Dataframe was not created correctly. Check dataset does not include infinite")
        train_frame = self.frame.copy(where= lambda row: row.Int <= 499990)
        ts_data = train_frame.take(n=train_frame.count(), columns="output2")
        ts_list = [item for sublist in ts_data for item in sublist]
        model = self.context.models.timeseries.arima.train(ts_list, 1, 1, 1, True)

        #These values are copied from the arimax datagen file.
        self.assertAlmostEqual(model.coefficients[0], 1.335, delta= 0.01)
        self.assertAlmostEqual(model.coefficients[1], 0.542, delta= 0.002)
        self.assertAlmostEqual(model.coefficients[2], 0.266, delta= 0.001)

    def test_arimax_predict(self):
        """Test ARIMAx predict method"""
        self.assertEqual(500000, self.frame.count(), msg= "Dataframe was not created correctly. Check dataset does not include infinite")
        train_frame = self.frame.copy(where= lambda row: row.Int <= 499990)
        actual_data = self.frame.copy(where= lambda row: row.Int > 499990) #last 10 rows
        ts_column ="output1"
        x_columns = ["exo1", "exo2", "exo3"]

        model = self.context.models.timeseries.arimax.train(train_frame, ts_column, x_columns, 1, 1, 1, 0, True, True)
        predict_frame = model.predict(actual_data, ts_column, x_columns)
        predict_data = predict_frame.take(n=actual_data.count(), columns="predicted_y")
        prediction = [item for sublist in predict_data for item in sublist]

        expected_data = actual_data.take(n=actual_data.count(), columns=ts_column)
        expected_prediction = [item for sublist in expected_data for item in sublist]

        for i in range(2,9):
            self.assertAlmostEqual(prediction[i], expected_prediction[i], delta=1000)

if __name__ == "__main__":
    unittest.main()
