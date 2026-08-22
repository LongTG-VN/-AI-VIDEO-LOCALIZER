from datetime import datetime, timezone
from pathlib import Path

from app.models.project import Project, ProjectPatch


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self.root / f"{project_id}.json"

    def save(self, project: Project) -> Project:
        project.updated_at = datetime.now(timezone.utc)
        self._path(project.id).write_text(project.model_dump_json(indent=2), encoding="utf-8")
        return project

    def create(self, project: Project) -> Project:
        return self.save(project)

    def get(self, project_id: str) -> Project:
        path = self._path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                projects.append(Project.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def patch(self, project_id: str, patch: ProjectPatch) -> Project:
        project = self.get(project_id)
        payload = patch.model_dump(exclude_unset=True)
        for key, value in payload.items():
            setattr(project, key, value)
        return self.save(project)
