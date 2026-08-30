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

            <section className="visual-edit-card">
              <div className="card-title">
                <span>Visual Edit Composer</span>
                <small>Style · Dynamic Blur · Graphic Overlays</small>
              </div>
              <div className="visual-edit-body">
                <div className="form-row">
                  <label className="field-label">Visual Style:</label>
                  <div className="style-selector">
                    <button
                      type="button"
                      className={(project.visual_edit?.mode ?? 'clean') === 'clean' ? 'active' : ''}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'clean',
                          blur: project.visual_edit?.blur ?? { enabled: false, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: false, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: project.visual_edit?.preset ?? 'default',
                        }
                      })}
                    >
                      Clean (Inpaint)
                    </button>
                    <button
                      type="button"
                      className={project.visual_edit?.mode === 'patch_cover' ? 'active recommended' : 'recommended'}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'patch_cover',
                          blur: project.visual_edit?.blur ?? { enabled: false, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: true, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: 'shortform_reference',
                        }
                      })}
                    >
                      ★ Patch Cover (Recommended)
                    </button>
                    <button
                      type="button"
                      className={project.visual_edit?.mode === 'blur' ? 'active' : ''}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'blur',
                          blur: project.visual_edit?.blur ?? { enabled: true, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: false, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: project.visual_edit?.preset ?? 'default',
                        }
                      })}
                    >
                      Blur
                    </button>
                    <button
                      type="button"
                      className={project.visual_edit?.mode === 'blur_overlay' ? 'active' : ''}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'blur_overlay',
                          blur: project.visual_edit?.blur ?? { enabled: true, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: false, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: project.visual_edit?.preset ?? 'default',
                        }
                      })}
                    >
                      Blur + Overlay
                    </button>
                  </div>
                </div>

                {project.visual_edit?.mode === 'patch_cover' && (
                  <div className="sliders-grid">
                    <div className="slider-item">
                      <label>Cover Strength ({Math.round((project.visual_edit?.patch_cover?.patch_opacity ?? 0.92) * 100)}%)</label>
                      <input
                        type="range"
                        min="0.70"
                        max="1.00"
                        step="0.01"
                        value={project.visual_edit?.patch_cover?.patch_opacity ?? 0.92}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              patch_cover: { ...project.visual_edit!.patch_cover!, patch_opacity: val }
                            }
                          })
                        }}
                      />
                    </div>
                    <div className="slider-item">
                      <label>Feather ({project.visual_edit?.patch_cover?.feather_px ?? 8}px)</label>
                      <input
                        type="range"
                        min="2"
                        max="20"
                        value={project.visual_edit?.patch_cover?.feather_px ?? 8}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              patch_cover: { ...project.visual_edit!.patch_cover!, feather_px: val }
                            }
                          })
                        }}
                      />
                    </div>
                    <div className="slider-item">
                      <label>Padding ({project.visual_edit?.patch_cover?.padding_px ?? 6}px)</label>
                      <input
                        type="range"
                        min="0"
                        max="16"
                        value={project.visual_edit?.patch_cover?.padding_px ?? 6}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              patch_cover: { ...project.visual_edit!.patch_cover!, padding_px: val }
                            }
                          })
                        }}
                      />
                    </div>
                    <div className="slider-item">
                      <label>Blur Softness (σ: {project.visual_edit?.patch_cover?.blur_sigma ?? 6})</label>
                      <input
                        type="range"
                        min="2"
                        max="18"
                        value={project.visual_edit?.patch_cover?.blur_sigma ?? 6}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              patch_cover: { ...project.visual_edit!.patch_cover!, blur_sigma: val }
                            }
                          })
                        }}
                      />
                    </div>
                  </div>
                )}

                <div className="form-row">
                  <label className="field-label">Subtitle Typography & Preset:</label>
                  <div className="style-selector">
                    <button
                      type="button"
                      className={(project.visual_edit?.preset ?? 'default') === 'default' ? 'active' : ''}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: project.visual_edit?.mode ?? 'clean',
                          blur: project.visual_edit?.blur ?? { enabled: false, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: false, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          subtitle_backing: { enabled: false, opacity: 0.60, padding_x: 18, padding_y: 8, corner_radius: 10, blur_radius: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: 'default',
                        }
                      })}
                    >
                      Classic Movie
                    </button>
                    <button
                      type="button"
                      className={((project.visual_edit?.preset ?? '') === 'shortform_white_black_soft_bg' || (project.visual_edit?.preset ?? '') === 'shortform_soft_bg') ? 'active recommended' : 'recommended'}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'patch_cover',
                          blur: project.visual_edit?.blur ?? { enabled: false, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: true, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          subtitle_backing: { enabled: true, opacity: 0.72, padding_x: 20, padding_y: 10, corner_radius: 10, blur_radius: 8 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: 'shortform_white_black_soft_bg',
                        }
                      })}
                    >
                      ★ Shortform Soft BG (White + Thin Black)
                    </button>
                    <button
                      type="button"
                      className={project.visual_edit?.preset === 'shortform_bold_yellow' ? 'active' : ''}
                      onClick={() => setProject({
                        ...project,
                        visual_edit: {
                          mode: 'patch_cover',
                          blur: project.visual_edit?.blur ?? { enabled: false, sigma: 18, padding_px: 8, feather_px: 6 },
                          patch_cover: project.visual_edit?.patch_cover ?? { enabled: true, patch_opacity: 0.92, padding_px: 6, feather_px: 8, blur_sigma: 6 },
                          subtitle_backing: { enabled: true, opacity: 0.72, padding_x: 20, padding_y: 10, corner_radius: 10, blur_radius: 6 },
                          overlays: project.visual_edit?.overlays ?? [],
                          preset: 'shortform_bold_yellow',
                        }
                      })}
                    >
                      Shortform Bold Yellow
                    </button>
                  </div>
                </div>

                {project.visual_edit?.subtitle_backing?.enabled && (
                  <div className="sliders-grid">
                    <div className="slider-item">
                      <label>Backing Opacity ({Math.round((project.visual_edit?.subtitle_backing?.opacity ?? 0.72) * 100)}%)</label>
                      <input
                        type="range"
                        min="0.30"
                        max="0.95"
                        step="0.05"
                        value={project.visual_edit?.subtitle_backing?.opacity ?? 0.72}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              subtitle_backing: { ...project.visual_edit!.subtitle_backing!, opacity: val }
                            }
                          })
                        }}
                      />
                    </div>
                    <div className="slider-item">
                      <label>Backing Padding ({project.visual_edit?.subtitle_backing?.padding_x ?? 18}px)</label>
                      <input
                        type="range"
                        min="4"
                        max="36"
                        value={project.visual_edit?.subtitle_backing?.padding_x ?? 18}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              subtitle_backing: { ...project.visual_edit!.subtitle_backing!, padding_x: val }
                            }
                          })
                        }}
                      />
                    </div>
                  </div>
                )}

                {project.visual_edit?.mode === 'blur_overlay' && (
                  <div className="overlays-section">
                    <div className="overlays-header">
                      <span>Image & Sticker Overlays</span>
                      <button
                        type="button"
                        className="ghost small"
                        onClick={() => {
                          const newOv = {
                            id: `ov_${Date.now()}`,
                            path: 'data/uploads/sample_sticker.png',
                            start: 0,
                            end: project.duration ? Math.min(10, project.duration) : 10,
                            x: 0.5,
                            y: 0.2,
                            width: 0.25,
                            opacity: 1.0,
                            fade_in_ms: 200,
                            fade_out_ms: 200,
                            z_index: 10,
                            anchor: 'absolute' as const,
                          }
                          setProject({
                            ...project,
                            visual_edit: {
                              ...project.visual_edit!,
                              overlays: [...(project.visual_edit!.overlays || []), newOv]
                            }
                          })
                        }}
                      >
                        + Add Overlay
                      </button>
                    </div>

                    {(project.visual_edit.overlays || []).length === 0 ? (
                      <div className="no-overlays">No graphic overlays added yet. Click &quot;+ Add Overlay&quot; to place images.</div>
                    ) : (
                      <div className="overlays-list">
                        {project.visual_edit.overlays.map((ov, idx) => (
                          <div className="overlay-row" key={ov.id}>
                            <input
                              type="text"
                              className="ov-path"
                              placeholder="Image path..."
                              value={ov.path}
                              onChange={(e) => {
                                const val = e.target.value
                                setProject({
                                  ...project,
                                  visual_edit: {
                                    ...project.visual_edit!,
                                    overlays: project.visual_edit!.overlays.map((item, i) => i === idx ? { ...item, path: val } : item)
                                  }
                                })
                              }}
                            />
                            <div className="ov-field">
                              <span>Time (s)</span>
                              <input
                                type="number"
                                step="0.5"
                                value={ov.start}
                                onChange={(e) => {
                                  const val = Number(e.target.value)
                                  setProject({
                                    ...project,
                                    visual_edit: {
                                      ...project.visual_edit!,
                                      overlays: project.visual_edit!.overlays.map((item, i) => i === idx ? { ...item, start: val } : item)
                                    }
                                  })
                                }}
                              />
                              <span>-</span>
                              <input
                                type="number"
                                step="0.5"
                                value={ov.end}
                                onChange={(e) => {
                                  const val = Number(e.target.value)
                                  setProject({
                                    ...project,
                                    visual_edit: {
                                      ...project.visual_edit!,
                                      overlays: project.visual_edit!.overlays.map((item, i) => i === idx ? { ...item, end: val } : item)
                                    }
                                  })
                                }}
                              />
                            </div>
                            <div className="ov-field">
                              <span>Scale</span>
                              <input
                                type="number"
                                step="0.05"
                                min="0.05"
                                max="1.0"
                                value={ov.width}
                                onChange={(e) => {
                                  const val = Number(e.target.value)
                                  setProject({
                                    ...project,
                                    visual_edit: {
                                      ...project.visual_edit!,
                                      overlays: project.visual_edit!.overlays.map((item, i) => i === idx ? { ...item, width: val } : item)
                                    }
                                  })
                                }}
                              />
                            </div>
                            <div className="ov-field">
                              <span>Opacity</span>
                              <input
                                type="number"
                                step="0.1"
                                min="0"
                                max="1"
                                value={ov.opacity}
                                onChange={(e) => {
                                  const val = Number(e.target.value)
                                  setProject({
                                    ...project,
                                    visual_edit: {
                                      ...project.visual_edit!,
                                      overlays: project.visual_edit!.overlays.map((item, i) => i === idx ? { ...item, opacity: val } : item)
                                    }
                                  })
                                }}
                              />
                            </div>
                            <button
                              type="button"
                              className="ghost remove-btn"
                              onClick={() => {
                                setProject({
                                  ...project,
                                  visual_edit: {
                                    ...project.visual_edit!,
                                    overlays: project.visual_edit!.overlays.filter((_, i) => i !== idx)
                                  }
                                })
                              }}
                            >
                              ✕
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
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
