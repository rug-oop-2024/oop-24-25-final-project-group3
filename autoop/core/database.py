import json
from typing import Tuple, List, Union
import pydoc  # noqa: F401

from autoop.core.storage import Storage


class Database():
    """
    Database class for managing collections of data with storage integration.

    This class provides methods for setting, getting, deleting, and listing
    data within collections stored in memory and persisted in a specified
    storage system. It ensures data is synchronized between in-memory
    representation and external storage, supporting operations such as refresh,
    persistence, and loading.

    Attributes:
        _storage (Storage): The storage backend used for persisting data.
        _data (dict): The in-memory representation of collections and their
                      entries.

    Methods:
        __init__(storage: Storage) -> None:
            Initializes the Database instance with a storage backend and loads
            data.

        set(collection: str, id: str, entry: dict) -> dict:
            Stores an entry in the specified collection with a given ID.

        get(collection: str, id: str) -> Union[dict, None]:
            Retrieves an entry from the specified collection by ID.

        delete(collection: str, id: str) -> None:
            Deletes an entry from the specified collection by ID.

        list(collection: str) -> List[Tuple[str, dict]]:
            Lists all entries in a specified collection.

        refresh() -> None:
            Reloads the in-memory database from the storage system.

        _persist() -> None:
            Persists the current state of the in-memory database to the
            storage,
            and deletes obsolete entries from storage.

        _load() -> None:
            Loads data from the storage into the in-memory database.
    """

    def __init__(self, storage: Storage) -> None:
        """Initialising Database"""
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Set a key in the database
        Args:
            collection (str): The collection to store the data in
            id (str): The id of the data
            entry (dict): The data to store
        Returns:
            dict: The data that was stored
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """
        Get a key from the database
        Args:
            collection (str): The collection to get the data from
            id (str): The id of the data
        Returns:
            Union[dict, None]: The data that was stored,
            or None if it doesn't exist
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """Delete a key from the database with debug logging
        Args:
            collection (str): The collection to delete the data from
            id (str): The id of the data
        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if not self._data[collection].get(id, None):
            return
        del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        Lists all data in a collection
        Args:
            collection (str): The collection to list the data from
        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the id and
            data for each item in the collection
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Refresh the database by loading the data from storage"""
        self._load()

    def _persist(self) -> None:
        """Persist the data to storage."""
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                # Save data to storage if it exists in memory
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}/{id}")

        # Check for items in storage that are no longer in memory
        keys = self._storage.list("")  # List all files in storage
        for key in keys:
            collection, id = key.split("/")[-2:]
            # Remove from storage if not in the in-memory database
            if not self._data.get(collection, {}).get(id):
                self._storage.delete(f"{collection}/{id}")

    def _load(self) -> None:
        """Load the data from storage"""
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split("/")[-2:]
            data = self._storage.load(f"{collection}/{id}")
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())

# pydoc.writedoc('database')
