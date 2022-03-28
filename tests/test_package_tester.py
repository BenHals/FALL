""" Simple tests to test CI pipeline. """
from streamselect.package_tester import run_package


def test_run_package() -> None:
    """Simple test to test package code."""
    assert run_package() == "Hello World"
