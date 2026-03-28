"""CLI exception classes with exit codes."""

import click


class CLIError(click.ClickException):
    """Base class for all CLI errors with exit codes."""

    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


class ParameterError(CLIError):
    """Command line parameter error (exit code 1)."""

    def __init__(self, message):
        super().__init__(message, exit_code=1)


class InputError(CLIError):
    """Input file/directory error (exit code 2)."""

    def __init__(self, message):
        super().__init__(message, exit_code=2)


class OutputError(CLIError):
    """Output file/directory error (exit code 3)."""

    def __init__(self, message):
        super().__init__(message, exit_code=3)


class RuntimeCLIError(CLIError):
    """Runtime error during API execution (exit code 4)."""

    def __init__(self, message):
        super().__init__(message, exit_code=4)


class SystemError(CLIError):
    """System-level error (exit code 5)."""

    def __init__(self, message):
        super().__init__(message, exit_code=5)