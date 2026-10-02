"""Provide unittest-style assertions for the plain-class terrain tests."""

import unittest


def pytest_collection_modifyitems(items):
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is not None and not hasattr(cls, "assertIsNotNone"):
            cls.assertIsNotNone = unittest.TestCase.assertIsNotNone
