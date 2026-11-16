"""Never send real feedback from the test suite (local runs or GitHub Actions)."""
import os

os.environ["METAPY_FEEDBACK"] = '{"action": ""}'
