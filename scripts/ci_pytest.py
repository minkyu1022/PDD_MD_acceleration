"""Run pytest and surface failures as public GitHub check annotations."""

import subprocess
import sys
import xml.etree.ElementTree as ET


def annotation(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-m", "not slow", "--junitxml=pytest.xml"],
    check=False,
)
if result.returncode:
    try:
        root = ET.parse("pytest.xml").getroot()
        for case in root.iter("testcase"):
            failure = case.find("failure")
            if failure is None:
                failure = case.find("error")
            if failure is not None:
                title = annotation(case.get("name", "pytest failure"))
                details = annotation(
                    (failure.text or failure.get("message", ""))[:3000]
                )
                print(f"::error title={title}::{details}", flush=True)
    except (OSError, ET.ParseError):
        pass
sys.exit(result.returncode)
