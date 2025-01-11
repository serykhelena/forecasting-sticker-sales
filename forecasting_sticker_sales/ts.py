from datetime import datetime

_DEFAULT_DATE_FORMAT = "%Y-%m-%d"
_DEFAULT_TIME_FORMAT = "%H-%M-%S"


def get_date_string(
    output_format: str | None = None,
    with_time: bool = False,
) -> str:
    """Get string representation of the current date(time)."""
    if output_format is None:
        output_format = _DEFAULT_DATE_FORMAT

        if with_time:
            output_format = f"{output_format}-{_DEFAULT_TIME_FORMAT}"

    return datetime.today().strftime(output_format)  # noqa: DTZ002
