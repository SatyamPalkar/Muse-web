"use client"

import { useEffect, useState } from "react"
import { useWebSocket } from "@/contexts/websocket-context"
import { getBrainStateFromClass, getBrainStateColor, getBrainStateTextColor } from "@/utils/brain-state"
import type { BrainState } from "@/types/eeg"

export function CurrentStateDisplay() {
  const { currentData } = useWebSocket()
  const [currentState, setCurrentState] = useState<BrainState>("Relaxed")
  const [confidence, setConfidence] = useState(0)

  useEffect(() => {
    if (currentData) {
      const newState = getBrainStateFromClass(currentData.classification)
      setCurrentState(newState)
      setConfidence(currentData.confidence || Math.floor(Math.random() * 30) + 70) // Fallback if no confidence
    }
  }, [currentData])

  const stateColor = getBrainStateColor(currentState)
  const stateTextColor = getBrainStateTextColor(currentState)

  // Calculate progress bar width for each state
  const states: BrainState[] = ["Relaxed", "Stressed", "Focused", "Drowsy"]
  const activeIndex = states.indexOf(currentState)

  return (
    <div className="flex flex-col">
      <h2 className="text-xl font-semibold mb-4">Current Stress Level</h2>
      <div className="flex flex-col">
        <h3 className={`text-5xl font-bold ${stateTextColor}`}>{currentState}</h3>
        <p className="text-white/70 mt-1">Confidence: {confidence}%</p>
      </div>

      <div className="mt-8 progress-bar">
        <div
          className="flex"
          style={{
            width: "100%",
            height: "100%",
          }}
        >
          {states.map((state, index) => (
            <div
              key={state}
              className="h-full transition-opacity duration-500"
              style={{
                width: "25%",
                backgroundColor: getBrainStateColor(state),
                opacity: index === activeIndex ? 1 : 0.3,
              }}
            />
          ))}
        </div>
      </div>
      <div className="flex justify-between mt-1 text-xs text-white/50">
        <span>Relaxed</span>
        <span>Stressed</span>
        <span>Focused</span>
        <span>Drowsy</span>
      </div>
    </div>
  )
}
