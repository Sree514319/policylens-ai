"""Typed application exceptions mapped to safe HTTP responses.

Every exception carries a client-safe ``detail`` message only. Internal
details (file paths, stack traces, parser internals) must never be placed
in ``detail`` since it is returned directly to the caller.
"""


class PolicyLensError(Exception):
    """Base class for all expected, handled application errors."""

    status_code: int = 400

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidFileExtensionError(PolicyLensError):
    status_code = 400


class InvalidContentTypeError(PolicyLensError):
    status_code = 400


class InvalidFileSignatureError(PolicyLensError):
    status_code = 400


class EmptyFileError(PolicyLensError):
    status_code = 400


class FileTooLargeError(PolicyLensError):
    status_code = 413


class CorruptedPDFError(PolicyLensError):
    status_code = 422


class EncryptedPDFError(PolicyLensError):
    status_code = 422
