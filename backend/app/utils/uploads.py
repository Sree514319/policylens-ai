"""Transport-level helpers for handling uploaded files safely."""

from fastapi import UploadFile

from app.core.exceptions import FileTooLargeError

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def read_upload_within_limit(upload_file: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile into memory, aborting as soon as it exceeds max_bytes.

    Reading in bounded chunks (rather than calling ``.read()`` once) avoids
    buffering an arbitrarily large payload in memory before rejecting it,
    which limits the impact of oversized-upload denial-of-service attempts.
    """

    chunks = []
    total_read = 0

    while True:
        chunk = await upload_file.read(_CHUNK_SIZE)
        if not chunk:
            break

        total_read += len(chunk)
        if total_read > max_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum allowed upload size of {max_bytes} bytes."
            )

        chunks.append(chunk)

    return b"".join(chunks)
