from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob
import pydoc  # noqa: F401


class NotFoundError(Exception):
    """
    Custom exception for handling cases where a path is not found.

    Args:
        path (str): The path that was not found.
    """
    def __init__(self, path: str) -> None:
        """Initialises the NotFoundError class"""
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for a storage interface defining methods for saving,
    loading, deleting, and listing data.
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Concrete implementation of the Storage interface using the local file
    system.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """Initialising the LocalStorage class"""
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to be saved.
            path (str): The path where data should be saved.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): The path to load data from.

        Returns:
            bytes: The data loaded from the given path.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a given path.

        Args:
            path (str): The path to delete data from.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): The path to list data from.

        Returns:
            list: A list of paths under the given path.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a given path exists and raise an error if not.

        Args:
            path (str): The path to check.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a given path.

        Args:
            path (str): The path to join with the base path.

        Returns:
            str: The combined path.
        """
        return os.path.join(self._base_path, path)

# pydoc.writedoc('storage')
