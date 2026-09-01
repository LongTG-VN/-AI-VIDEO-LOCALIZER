export type SubtitleCue = {
  id: string
  start: number
  end: number
  speaker_id?: string | null
  speaker_character_id?: string | null
  addressee_id?: string | null
  addressee_character_id?: string | null
  source_text: string
  translated_text?: string | null
  confidence?: number | null
  translation_confidence?: number | null
  relationship_confidence?: number | null
  needs_review?: boolean
  review_notes?: string | null
  critic_score?: number | null
  critic_flags?: string[]
  ocr_start?: number | null
  ocr_end?: number | null
  ocr_text?: string | null
  ocr_regions?: OCRRegion[]
}

export type OCRRegion = {
  text?: string | null
  confidence?: number | null
  points: number[][]
}

export type Character = {
  id: string
  name: string
  name_zh?: string | null
  name_vi?: string | null
  aliases?: string[]
  gender?: string | null
  role?: string | null
  description?: string | null
  speaker_ids?: string[]
  confidence?: number | null
  notes?: string | null
}

export type Relationship = {
  id: string
  from_character_id: string
  to_character_id: string
  relationship: string
  relationship_type?: string | null
  valid_from: number
  valid_until?: number | null
  vi_self?: string | null
  vi_other?: string | null
  vi_self_pronoun?: string | null
  vi_target_pronoun?: string | null
  en_register?: string | null
  confidence?: number | null
  notes?: string | null
}

export type GlossaryEntry = {
  id: string
  source: string
  target: string
  category?: string | null
  confidence?: number | null
  note?: string | null
}

export type Scene = {
  id: string
  scene_id?: string | null
  start: number
  end: number
  summary: string
  tone?: string | null
  characters: string[]
}

export type VisualEditMode = 'clean' | 'patch_cover' | 'blur' | 'blur_overlay'

export type OverlayAnchor = 'absolute' | 'subtitle_region' | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right' | 'center'

export type BlurConfig = {
  enabled: boolean
  sigma: number
  padding_px: number
  feather_px: number
  min_ocr_confidence?: number
  temporal_gap_fill_frames?: number
}

export type PatchCoverConfig = {
  enabled: boolean
  patch_opacity: number
  padding_px: number
  feather_px: number
  blur_sigma: number
  min_ocr_confidence?: number
  temporal_gap_fill_frames?: number
  mask_persistence_frames?: number
  use_temporal_donor?: boolean
  use_spatial_donor?: boolean
}

export type OverlayConfig = {
  id: string
  path: string
  start: number
  end: number
  x: number
  y: number
  width: number
  opacity: number
  fade_in_ms?: number
  fade_out_ms?: number
  z_index?: number
  anchor?: OverlayAnchor
}

export type SubtitleBackingConfig = {
  enabled: boolean
  color?: string
  opacity: number
  padding_x: number
  padding_y: number
  corner_radius: number
  blur_radius: number
}

export type VisualEditConfig = {
  mode: VisualEditMode
  blur: BlurConfig
  patch_cover?: PatchCoverConfig
  subtitle_backing?: SubtitleBackingConfig
  overlays: OverlayConfig[]
  preset?: 'default' | 'shortform_reference' | 'shortform_bold_yellow' | 'shortform_white_black_soft_bg' | 'shortform_soft_bg'
}

export type Project = {
  id: string
  name: string
  source_video_path: string
  source_language?: string
  target_language: 'vi' | 'en'
  duration?: number | null
  width?: number | null
  height?: number | null
  scenes?: Scene[]
  characters: Character[]
  relationships: Relationship[]
  glossary?: GlossaryEntry[]
  cues: SubtitleCue[]
  visual_edit?: VisualEditConfig | null
}

