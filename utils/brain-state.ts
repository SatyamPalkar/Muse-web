import type { BrainState } from "@/types/eeg"

export const getBrainStateFromClass = (classification: number): BrainState => {
  switch (classification) {
    case 0:
      return "Relaxed"
    case 1:
      return "Stressed"
    case 2:
      return "Focused"
    case 3:
      return "Drowsy"
    default:
      return "Relaxed"
  }
}

export const getBrainStateColor = (state: BrainState): string => {
  switch (state) {
    case "Relaxed":
      return "#00ff9d" // Bright green
    case "Stressed":
      return "#ff3a5e" // Bright red
    case "Focused":
      return "#3a85ff" // Bright blue
    case "Drowsy":
      return "#b467ff" // Bright purple
    default:
      return "#ffffff"
  }
}

export const getBrainStateTextColor = (state: BrainState): string => {
  switch (state) {
    case "Relaxed":
      return "text-[#00ff9d]"
    case "Stressed":
      return "text-[#ff3a5e]"
    case "Focused":
      return "text-[#3a85ff]"
    case "Drowsy":
      return "text-[#b467ff]"
    default:
      return "text-white"
  }
}

export const getBrainStateDescription = (state: BrainState): string => {
  switch (state) {
    case "Relaxed":
      return "Calm mental state with reduced neural activity"
    case "Stressed":
      return "Elevated neural activity indicating mental strain"
    case "Focused":
      return "Directed attention with organized neural patterns"
    case "Drowsy":
      return "Reduced alertness with slowed neural activity"
    default:
      return ""
  }
}

export const getBrainwaveColor = (band: string): string => {
  switch (band.toLowerCase()) {
    case "alpha":
      return "#00ff9d" // Green
    case "beta":
      return "#3a85ff" // Blue
    case "theta":
      return "#b467ff" // Purple
    case "delta":
      return "#ff3a5e" // Pink
    case "gamma":
      return "#ff9f1a" // Orange
    default:
      return "#ffffff" // White
  }
}
