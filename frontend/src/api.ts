import type { Project } from './types'

async function decode<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? 'Request failed')
  }
  return response.json() as Promise<T>
}

export async function listProjects(): Promise<Project[]> { return decode(await fetch('/api/projects')) }

export async function importVideo(file: File, targetLanguage: 'vi' | 'en'): Promise<Project> {
  const body = new FormData()
  body.append('file', file)
  body.append('name', file.name.replace(/\.[^.]+$/, ''))
  body.append('target_language', targetLanguage)
  return decode(await fetch('/api/projects/import', { method: 'POST', body }))
}

export async function analyze(projectId: string): Promise<Project> { return decode(await fetch(`/api/projects/${projectId}/analyze`, { method: 'POST' })) }
export async function transcribe(projectId: string): Promise<Project> { return decode(await fetch(`/api/projects/${projectId}/transcribe`, { method: 'POST' })) }
export async function inferContext(projectId: string): Promise<Project> { return decode(await fetch(`/api/projects/${projectId}/context`, { method: 'POST' })) }
export async function translate(projectId: string): Promise<Project> { return decode(await fetch(`/api/projects/${projectId}/translate`, { method: 'POST' })) }

export async function saveProject(project: Project): Promise<Project> {
  return decode(await fetch(`/api/projects/${project.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target_language: project.target_language, characters: project.characters, relationships: project.relationships, cues: project.cues }) }))
}

export async function renderProject(projectId: string): Promise<{ download_url: string }> {
  return decode(await fetch(`/api/projects/${projectId}/render`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ stickers: [] }) }))
}
