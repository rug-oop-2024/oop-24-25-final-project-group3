from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List
import pydoc


class ArtifactRegistry():
    """
    Manages artifact registration, retrieval, listing, and deletion by
    interacting with a database and storage system.
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initialize the ArtifactRegistry with a database and storage system.

        Args:
            database (Database): The database for artifact metadata.
            storage (Storage): The storage system for artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact by saving its data and metadata.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        List all artifacts, optionally filtering by type.

        Args:
            type (str, optional): Filter for artifact type.

        Returns:
            List[Artifact]: A list of Artifact objects.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to retrieve.

        Returns:
            Artifact: The retrieved Artifact object.
        """        """
        Retrieve an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to retrieve.

        Returns:
            Artifact: The retrieved Artifact object.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to delete.

        Raises:
            ValueError: If the artifact cannot be found or deleted.
        """
        # Retrieve artifact metadata to get the asset path
        data = self._database.get("artifacts", artifact_id)
        if data is None:
            raise ValueError(f"Artifact with ID '{artifact_id}' does not "
                             "exist.")

        asset_path = data["asset_path"]
        try:
            self._storage.delete(asset_path)
            # Delete metadata from the database
            self._database.delete("artifacts", artifact_id)
        except Exception as e:
            raise ValueError(f"Failed to delete artifact: {e}")


class AutoMLSystem:
    """
    A singleton class for managing an AutoML system with storage, database,
    and artifact registry.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoMLSystem with storage and a database.

        Args:
            storage (LocalStorage): The storage system.
            database (Database): The database for storing metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Get the singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> 'ArtifactRegistry':
        """
        Access the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry.
        """
        return self._registry


if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)
