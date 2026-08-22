export type SubtitleCue = {
  id: string
  start: number
  end: number
  speaker_id?: string | null
  addressee_id?: string | null
  source_text: string
  translated_text?: string | null
  confidence?: number | null
}

export type Character = {
  id: string
  name: string
  aliases: string[]
  gender?: string | null
  role?: string | null
  notes?: string | null
}

export type Relationship = {
  id: string
  from_character_id: string
  to_character_id: string
  relationship: string
  valid_from: number
  valid_until?: number | null
  vi_self?: string | null
  vi_other?: string | null
  en_register?: string | null
}

export type Project = {
  id: string
  name: string
  source_video_path: string
  target_language: 'vi' | 'en'
  duration?: number | null
  width?: number | null
  height?: number | null
  characters: Character[]
  relationships: Relationship[]
  cues: SubtitleCue[]
}
