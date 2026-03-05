from numereng.platform.errors import ForumScraperError, NumeraiClientError, PackageError


def test_package_error_is_exception() -> None:
    assert issubclass(PackageError, Exception)


def test_numerai_client_error_is_exception() -> None:
    assert issubclass(NumeraiClientError, PackageError)


def test_forum_scraper_error_is_exception() -> None:
    assert issubclass(ForumScraperError, PackageError)


def test_forum_scraper_error_message_round_trips() -> None:
    error = ForumScraperError("forum_scraper_network_error")
    assert str(error) == "forum_scraper_network_error"
