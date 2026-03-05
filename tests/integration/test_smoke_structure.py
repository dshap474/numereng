import pytest

from numereng import cli
from numereng.api import get_health


@pytest.mark.integration
def test_api_cli_smoke_contract(capsys: pytest.CaptureFixture[str]) -> None:
    response = get_health()
    exit_code = cli.main([])
    _ = capsys.readouterr()

    assert response.status == "ok"
    assert response.package == "numereng"
    assert exit_code == 0
