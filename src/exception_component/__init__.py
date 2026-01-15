import traceback
from typing import Optional

class MyException(Exception):
    """
    Custom exception with optional original exception context and full traceback support.
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        # Save the custom message and original exception
        self.message = message
        self.original_exception = original_exception
        super().__init__(message)

    def __str__(self) -> str:
        if self.original_exception:
            return f"{self.message} | Original: {repr(self.original_exception)}"
        return self.message

    def full_traceback(self) -> str:
        """
        Returns the full traceback of the original exception as a string.
        If no original exception exists, returns an empty string.
        """
        if not self.original_exception:
            return ""
        return "".join(traceback.format_exception(
            type(self.original_exception),
            self.original_exception,
            self.original_exception.__traceback__
        ))
