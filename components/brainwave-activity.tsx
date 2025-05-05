"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { useEffect, useState } from "react"
import { getBrainwaveColor } from "@/utils/brain-state"

export function BrainwaveActivity() {
  const { currentData } = useWebSocket()
  const [waveData, setWaveData] = useState<Record<string, number[]>>({
    alpha: Array(100).fill(0),
    beta: Array(100).fill(0),
    theta: Array(100).fill(0),
    delta: Array(100).fill(0),
    gamma: Array(100).fill(0),
  })

  useEffect(() => {
    if (currentData) {
      // Update wave data with the latest values
      setWaveData((prev) => {
        const newData = { ...prev }

        // For each brainwave type, shift the array and add new value
        Object.keys(currentData.brainwaves).forEach((wave) => {
          const values = currentData.brainwaves[wave as keyof typeof currentData.brainwaves]
          const lastValue = values[values.length - 1] || 0
          newData[wave] = [...prev[wave].slice(1), lastValue]
        })

        return newData
      })
    } else {
      // Generate some fake wave data for demonstration
      const interval = setInterval(() => {
        setWaveData((prev) => {
          const newData = { ...prev }

          // Generate sine waves with different frequencies and amplitudes
          const time = Date.now() / 1000
          newData.alpha = [...prev.alpha.slice(1), Math.sin(time * 2) * 40 + 50]
          newData.beta = [...prev.beta.slice(1), Math.sin(time * 5) * 30 + 50]
          newData.theta = [...prev.theta.slice(1), Math.sin(time * 1.5) * 35 + 50]
          newData.delta = [...prev.delta.slice(1), Math.sin(time * 1) * 45 + 50]
          newData.gamma = [...prev.gamma.slice(1), Math.sin(time * 8) * 20 + Math.sin(time * 2) * 10 + 50]

          return newData
        })
      }, 50)

      return () => clearInterval(interval)
    }
  }, [currentData])

  // Create SVG paths for each wave
  const createPath = (data: number[], index: number, height = 80) => {
    const width = 600
    const points = data
      .map((value, i) => {
        const x = (i / (data.length - 1)) * width
        const y = height - (value / 100) * height + index * (height + 20)
        return `${x},${y}`
      })
      .join(" L ")

    return `M ${points}`
  }

  const waves = [
    { name: "Alpha", data: waveData.alpha, color: getBrainwaveColor("alpha") },
    { name: "Beta", data: waveData.beta, color: getBrainwaveColor("beta") },
    { name: "Theta", data: waveData.theta, color: getBrainwaveColor("theta") },
    { name: "Delta", data: waveData.delta, color: getBrainwaveColor("delta") },
    { name: "Raw EEG", data: waveData.gamma, color: getBrainwaveColor("gamma") },
  ]

  return (
    <div className="flex flex-col">
      <h2 className="text-xl font-semibold mb-4">Brainwave Activity</h2>
      <div className="relative h-[400px] w-full">
        <svg width="100%" height="100%" viewBox="0 0 600 400" preserveAspectRatio="none">
          {waves.map((wave, index) => (
            <g key={wave.name}>
              <text x="10" y={index * 80 + 40} fill={wave.color} fontSize="12" dominantBaseline="middle">
                {wave.name}
              </text>
              <path d={createPath(wave.data, index)} className="wave-line" style={{ stroke: wave.color }} />
            </g>
          ))}
        </svg>
      </div>
    </div>
  )
}
