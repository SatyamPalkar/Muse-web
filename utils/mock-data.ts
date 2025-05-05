import type { EEGData } from "@/types/eeg"

// Function to generate random EEG data for testing
export function generateMockEEGData(): EEGData {
  const timestamp = Date.now()

  // Generate random classification (0-3)
  const classification = Math.floor(Math.random() * 4)

  // Generate random brainwave data
  const generateWave = (length: number) => {
    return Array.from({ length }, () => Math.random() * 100)
  }

  // Generate random features
  const features: Record<string, number> = {
    mean_alpha: Math.random() * 50,
    mean_beta: Math.random() * 50,
    mean_theta: Math.random() * 50,
    mean_delta: Math.random() * 50,
    mean_gamma: Math.random() * 50,
    std_alpha: Math.random() * 10,
    std_beta: Math.random() * 10,
    std_theta: Math.random() * 10,
    std_delta: Math.random() * 10,
    std_gamma: Math.random() * 10,
    alpha_beta_ratio: Math.random() * 2,
    theta_beta_ratio: Math.random() * 2,
    alpha_theta_ratio: Math.random() * 2,
    delta_theta_ratio: Math.random() * 2,
    gamma_beta_ratio: Math.random() * 2,
  }

  // Generate random attention weights
  const attentionWeights: Record<string, number> = {}
  Object.keys(features).forEach((key) => {
    attentionWeights[key] = Math.random()
  })

  // Normalize attention weights
  const sum = Object.values(attentionWeights).reduce((a, b) => a + b, 0)
  Object.keys(attentionWeights).forEach((key) => {
    attentionWeights[key] = attentionWeights[key] / sum
  })

  return {
    timestamp,
    rawValues: generateWave(100),
    brainwaves: {
      alpha: generateWave(50),
      beta: generateWave(50),
      theta: generateWave(50),
      delta: generateWave(50),
      gamma: generateWave(50),
    },
    features,
    classification,
    attentionWeights,
  }
}
