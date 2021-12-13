import os
import unittest
import tempfile
import time
from unittest.mock import patch, MagicMock

import pandas as pd

import dfhist
from dfhist import DFHist


TEST_DF = pd.DataFrame(
    {
        "ints": [1, 2, 3],
        "floats": [1.23, 4.56, 7.89],
        "bools": [True, False, True],
        "strs": ["cake", "ham", "eggs"],
    }
)


class TestDfhist(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)

        self.count = 0

        self.dfhist = DFHist(
            directory=self.td.name,
            expire=None,
            tsformatter=self.counter,
            method="csv",
        )

    def counter(self):
        """Use this for tsformatter"""
        self.count += 1
        return str(self.count)

    def assertIsFile(self, filename):
        self.assertTrue(os.path.isfile(os.path.join(self.td.name, filename)))

    def assertIsNotFile(self, filename):
        self.assertFalse(os.path.isfile(os.path.join(self.td.name, filename)))

    def test_csv_marshal_round_trip(self):
        path = self.dfhist.marshal(TEST_DF)
        self.assertTrue(os.path.isfile(path))
        restored_df = self.dfhist.unmarshal(path)
        self.assertTrue(TEST_DF.equals(restored_df))

    def test_handles_no_cache_yet(self):
        # given no cache file yet
        @self.dfhist
        def fn():
            return TEST_DF

        # when
        df = fn()

        # then
        self.assertTrue(TEST_DF.equals(df))
        self.assertTrue(os.path.isfile(os.path.join(self.td.name, "1.csv")))

    def test_uses_cache(self):
        m = MagicMock(return_value=TEST_DF)

        @self.dfhist
        def fn():
            return m()

        # when
        fn()
        # then
        m.assert_called()
        self.assertIsFile("1.csv")

        # when
        m.reset_mock()
        fn()
        # then
        m.assert_not_called()
        self.assertIsNotFile("2.csv")

    def test_cache_instant_expiry(self):
        m = MagicMock(return_value=TEST_DF)

        self.dfhist.expiry = 0

        @self.dfhist
        def fn():
            return m()

        # when
        df = fn()
        # then
        m.assert_called()
        self.assertTrue(TEST_DF.equals(df))
        self.assertIsFile("1.csv")

        # when
        m.reset_mock()
        df2 = fn()
        # then
        m.assert_called()
        self.assertTrue(TEST_DF.equals(df2))
        self.assertIsFile("2.csv")

    def test_cache_expiry(self):
        m = MagicMock(return_value=TEST_DF)

        self.dfhist.expiry = 1

        @self.dfhist
        def fn():
            return m()

        # when
        df = fn()
        # then
        m.assert_called()
        self.assertTrue(TEST_DF.equals(df))
        self.assertIsFile("1.csv")

        # when
        time.sleep(1.5)
        m.reset_mock()
        df2 = fn()
        # then
        m.assert_called()
        self.assertTrue(TEST_DF.equals(df2))
        self.assertIsFile("2.csv")

    def test_retrieve(self):
        # given
        self.dfhist.unmarshal = MagicMock(side_effect=self.dfhist.unmarshal)

        @self.dfhist
        def fn():
            return TEST_DF

        fn.force()

        # when
        df = fn()
        # then
        self.dfhist.unmarshal.assert_called()
        self.assertTrue(TEST_DF.equals(df))

    def test_force_rewrite(self):
        m = MagicMock(return_value=TEST_DF)

        @self.dfhist
        def fn():
            return m()

        # when
        fn()
        # then
        m.assert_called()
        self.assertIsFile("1.csv")

        # when
        m.reset_mock()
        fn.force()
        # then
        m.assert_called()
        self.assertIsFile("2.csv")
