from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    def __init__(self,
                 database: Database,
                 storage: Storage):
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
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

    def delete(self, artifact_id: str):
        # Retrieve artifact metadata to get the asset path
        data = self._database.get("artifacts", artifact_id)
        if data is None:
            raise ValueError(f"Artifact with ID '{artifact_id}' does not exist.")
        
        asset_path = data["asset_path"]
        try:
            # Attempt to delete the artifact's data from storage
            self._storage.delete(asset_path)
            print(f"Deleted artifact data at {asset_path} from storage.")

            # Delete metadata from the database
            self._database.delete("artifacts", artifact_id)
            print(f"Deleted artifact metadata with ID '{artifact_id}' from database.")

        except Exception as e:
            raise ValueError(f"Failed to delete artifact: {e}")



class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
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
    def registry(self):
        return self._registry
