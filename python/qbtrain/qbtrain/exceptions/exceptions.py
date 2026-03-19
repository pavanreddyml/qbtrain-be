# qbtrain/exceptions/exceptions.py


class PermissionError(Exception):
    """Exception raised for permission errors in AI agents."""
    pass


class DenylistViolationError(Exception):
    """Raised when generated code contains denied libraries or commands."""
    pass


class StudentModelTooLargeError(Exception):
    """Raised when a student model exceeds the maximum allowed parameter count."""
    pass