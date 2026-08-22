import { useEffect, useMemo, useState } from 'react'
import { Download, FileVideo, Languages, Play, Save, Sparkles, Upload, WandSparkles } from 'lucide-react'
import { analyze, importVideo, inferContext, listProjects, renderProject, saveProject, translate } from './api'
import type { Project } from './types'

function fmt(seconds?: number | null) {
  if (seconds == null) return '--:--'
  const min = Math.floor(seconds / 60)
  const sec = Math.floor(seconds % 60)
  return `${min}:${sec.toString().padStart(2, '0')}`
}

function confidenceClass(value?: number | null) {
  if (value == null) return 'unknown'
  if (value >= 0.9) return 'good'
  if (value >= 0.7) return 'warn'
  return 'bad'
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [project, setProject] = useState<Project | null>(null)
  const [language, setLanguage] = useState<'vi' | 'en'>('vi')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [downloadUrl, setDownloadUrl] = useState('')

  useEffect(() => {
    listProjects().then((items) => {
      setProjects(items)
      if (items[0]) setProject(items[0])
    }).catch(() => undefined)
  }, [])

  const stats = useMemo(() => {
    if (!project) return { total: 0, review: 0 }
    return {
      total: project.cues.length,
      review: project.cues.filter((cue) => (cue.confidence ?? 1) < 0.7).length,
    }
  }, [project])

  async function run(label: string, fn: () => Promise<Project>) {
    setBusy(label)
    setError('')
    try {
      const updated = await fn()
      setProject(updated)
      setProjects((items) => [updated, ...items.filter((item) => item.id !== updated.id)])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy('')
    }
  }

  async function handleFile(file?: File) {
    if (!file) return
    await run('Importing video…', () => importVideo(file, language))
  }

  async function handleRender() {
    if (!project) return
    setBusy('Rendering video…')
    setError('')
    try {
      const result = await renderProject(project.id)
      setDownloadUrl(result.download_url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="app-shell">
      <header>
        <div className="brand"><Sparkles size={19} /> AI Video Localizer <span>alpha</span></div>
        <div className="header-actions">
          {busy && <span className="busy">{busy}</span>}
          {project && <button className="ghost" onClick={() => run('Saving…', () => saveProject(project))}><Save size={16}/> Save</button>}
        </div>
      </header>

      <aside>
        <div className="section-title">Projects</div>
        <label className="upload-button">
          <Upload size={17}/> New video
          <input type="file" accept="video/*" onChange={(event) => handleFile(event.target.files?.[0])} />
        </label>
        <div className="lang-row">
          <button className={language === 'vi' ? 'selected' : ''} onClick={() => setLanguage('vi')}>VI</button>
          <button className={language === 'en' ? 'selected' : ''} onClick={() => setLanguage('en')}>EN</button>
        </div>
        <div className="project-list">
          {projects.map((item) => (
            <button key={item.id} className={project?.id === item.id ? 'project active' : 'project'} onClick={() => setProject(item)}>
              <FileVideo size={16}/><span>{item.name}</span>
            </button>
          ))}
        </div>
      </aside>

      <main>
        {!project ? (
          <div className="empty-state">
            <div className="drop-icon"><Languages size={34}/></div>
            <h1>Localize Chinese video</h1>
            <p>Import a video, analyze dialogue, preserve relationships, translate, review and render.</p>
            <label className="primary big"><Upload size={18}/> Select video<input type="file" accept="video/*" onChange={(event) => handleFile(event.target.files?.[0])}/></label>
          </div>
        ) : (
          <>
            <section className="project-heading">
              <div>
                <div className="eyebrow">CURRENT PROJECT</div>
                <h1>{project.name}</h1>
                <p>{fmt(project.duration)} · {project.width ?? '?'}×{project.height ?? '?'} · Chinese → {project.target_language === 'vi' ? 'Vietnamese' : 'English'}</p>
              </div>
              <div className="actions">
                <button className="secondary" onClick={() => run('Analyzing ASR + OCR…', () => analyze(project.id))}><WandSparkles size={16}/> Analyze</button>
                <button className="secondary" onClick={() => run('Inferring roles…', () => inferContext(project.id))}><Sparkles size={16}/> Infer roles</button>
                <button className="primary" onClick={() => run('Translating…', () => translate(project.id))}><Languages size={16}/> Translate</button>
                <button className="secondary" onClick={handleRender}><Play size={16}/> Render</button>
              </div>
            </section>

            {error && <div className="error-banner">{error}</div>}
            {downloadUrl && <a className="download" href={downloadUrl}><Download size={16}/> Download rendered MP4</a>}

            <section className="stats">
              <div><strong>{stats.total}</strong><span>Subtitle cues</span></div>
              <div><strong>{project.characters.length}</strong><span>Characters</span></div>
              <div><strong>{project.relationships.length}</strong><span>Relationship rules</span></div>
              <div><strong>{stats.review}</strong><span>Needs review</span></div>
            </section>

            <section className="editor-card">
              <div className="card-title"><span>Subtitle editor</span><small>Original / translated / confidence</small></div>
              {project.cues.length === 0 ? (
                <div className="no-cues">No cues yet. Run Analyze or import SRT through the API.</div>
              ) : (
                <div className="cue-list">
                  {project.cues.map((cue, index) => (
                    <div className="cue" key={cue.id}>
                      <div className="time">{fmt(cue.start)}<br/><span>#{index + 1}</span></div>
                      <div className="texts">
                        <div className="source">{cue.source_text}</div>
                        <textarea value={cue.translated_text ?? ''} placeholder="Translation…" onChange={(event) => setProject({...project, cues: project.cues.map((item) => item.id === cue.id ? {...item, translated_text: event.target.value} : item)})} />
                      </div>
                      <div className={`confidence ${confidenceClass(cue.confidence)}`}>{cue.confidence == null ? '—' : `${Math.round(cue.confidence * 100)}%`}</div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  )
}
