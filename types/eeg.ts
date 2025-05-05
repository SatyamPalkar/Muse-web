export type BrainState = "Relaxed" | "Stressed" | "Focused" | "Drowsy"

export type StressLevel = "Positive" | "Acute" | "Episodic" | "Toxic"

export interface BrainWaves {
  alpha: number[]
  beta: number[]
  theta: number[]
  delta: number[]
  gamma: number[]
}

export interface EEGData {
  timestamp: number
  rawValues: number[]
  brainwaves: BrainWaves
  features: Record<string, number>
  classification: number
  confidence: number
  attentionWeights: Record<string, number>
}

export interface HistoricalData {
  timestamp: number
  state: BrainState
  level?: StressLevel
}

export interface StateDistribution {
  state: BrainState
  timeSpent: number
  percentage: number
}

export interface BrainWaveDistribution {
  band: string
  power: number
  description: string
}

export interface SessionMetrics {
  startTime: number
  duration: number
  calmPeriods: number
  peakState: BrainState
  averageStressLevel?: number
  peakStressLevel?: StressLevel
  stressTransitions?: number
}
