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
}

