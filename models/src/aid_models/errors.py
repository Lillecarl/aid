"""The one error every aid client raises: a stable `code` a caller branches on, and a message for people."""


class AidError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
