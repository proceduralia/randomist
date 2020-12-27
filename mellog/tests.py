import unittest
from logger import Mellogger
import pandas as pd
import json
import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")


class TestLocalLogging(unittest.TestCase):
    def setUp(self):
        self.log_dir = "test_dir"
        self.exp_name = "test"

    def test_base_dir_content(self):
        logger = Mellogger(log_dir=self.log_dir, exp_name=self.exp_name, args={"a": 2})
        infopath = os.path.join(logger.base_dir, "system_info.json")
        paramspath = os.path.join(logger.base_dir, "parameters.json")
        self.assertTrue(os.path.exists(logger.base_dir))
        self.assertTrue(os.path.exists(infopath))
        self.assertTrue(os.path.exists(paramspath))

    def test_parameters_in_file(self):
        file = tempfile.NamedTemporaryFile('r+')
        # Write some dummy parameters into the file
        dummy = {'param1': 1, 'param2': [1, 2, 3], 'param3': "test"}
        json.dump(dummy, file)
        file.read()
        # Check whether the parameters are correctly read and registered
        logger = Mellogger(log_dir=self.log_dir, exp_name=self.exp_name, args=file.name)
        file.close()
        self.assertEqual(logger.parameters, dummy)

    def test_short_metric_logging(self):
        logger = Mellogger(log_dir=self.log_dir, exp_name=self.exp_name)
        logger.log("test1", 1.2)
        logger.log("test1", 1.3)
        logger.log("test1", 1.4)
        logger.dump_metrics()
        path = os.path.join(logger.base_dir, "metrics.csv")
        df = pd.read_csv(path)
        self.assertEqual(list(df['test1']), [1.2, 1.3, 1.4])

    def test_metric_dumping(self):
        dump_frequency = 100
        # Test automatic dumping of metrics
        logger = Mellogger(log_dir=self.log_dir, exp_name=self.exp_name, dump_frequency=dump_frequency)
        for i in range(1000):
            logger.log("test1", i)
            logger.log("test2", i+1)
            if i % (dump_frequency//2) == 0 and i > dump_frequency:
                # Check that dumping is updated and correct
                path = os.path.join(logger.base_dir, "metrics.csv")
                df = pd.read_csv(path)
                self.assertEqual(list(df['test1']), list(range(i)))
                self.assertEqual(list(df['test2']), list(j+1 for j in range(i)))

    def tearDown(self):
        # Clean directories after each test is executed
        shutil.rmtree(self.log_dir, ignore_errors=True)


class TestNeptuneLogging(unittest.TestCase):
    def setUp(self):
        self.log_dir = "test_dir"
        self.exp_name = "test"

    def test_metric_dumping(self):
        logger = Mellogger(log_dir=self.log_dir, exp_name=self.exp_name, args={"a": 2},
                           external_logging="neptune", external_project="sandbox",
                           external_account="bellamy442")
        for i in range(1000):
            logger.log("test1", i)
            logger.log("test2", i+1)

    def tearDown(self):
        # Clean directories after each test is executed
        shutil.rmtree(self.log_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
