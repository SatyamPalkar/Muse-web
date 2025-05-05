import type { StressLevel } from "@/types/eeg"

export const getStressLevelFromClass = (classification: number): StressLevel => {
  switch (classification) {
    case 0:
      return "Positive"
    case 1:
      return "Acute"
    case 2:
      return "Episodic"
    case 3:
      return "Toxic"
    default:
      return "Positive"
  }
}

export const getStressLevelColor = (level: StressLevel): string => {
  switch (level) {
    case "Positive":
      return "bg-emerald-500"
    case "Acute":
      return "bg-amber-500"
    case "Episodic":
      return "bg-orange-500"
    case "Toxic":
      return "bg-rose-500"
    default:
      return "bg-gray-500"
  }
}

export const getStressLevelTextColor = (level: StressLevel): string => {
  switch (level) {
    case "Positive":
      return "text-emerald-500"
    case "Acute":
      return "text-amber-500"
    case "Episodic":
      return "text-orange-500"
    case "Toxic":
      return "text-rose-500"
    default:
      return "text-gray-500"
  }
}

export const getStressLevelStrokeColor = (level: StressLevel): string => {
  switch (level) {
    case "Positive":
      return "#10b981"
    case "Acute":
      return "#f59e0b"
    case "Episodic":
      return "#f97316"
    case "Toxic":
      return "#f43f5e"
    default:
      return "#6b7280"
  }
}

export const getStressLevelDescription = (level: StressLevel): string => {
  switch (level) {
    case "Positive":
      return "Healthy stress that motivates and improves performance"
    case "Acute":
      return "Short-term stress response to immediate challenges"
    case "Episodic":
      return "Frequent acute stress that may lead to health issues"
    case "Toxic":
      return "Chronic stress that can cause serious health problems"
    default:
      return ""
  }
}
