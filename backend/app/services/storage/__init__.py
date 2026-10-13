from app.services.storage.base import BaseStorage
from app.services.storage.local import LocalStorage

storage_service: BaseStorage = LocalStorage()

__all__ = ["BaseStorage", "LocalStorage", "storage_service"]
