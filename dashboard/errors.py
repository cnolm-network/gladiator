"""统一的 API 错误类型。"""

from __future__ import annotations


class ApiError(RuntimeError):
    """带 HTTP 状态码的可读错误。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message


class BadRequest(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(400, message)


class Forbidden(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(403, message)


class NotFound(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(404, message)


class Conflict(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(409, message)


class PayloadTooLarge(ApiError):
    def __init__(self, message: str = "请求体过大") -> None:
        super().__init__(413, message)
