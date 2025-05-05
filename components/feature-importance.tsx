"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { getBrainwaveColor } from "@/utils/brain-state"
import { useEffect, useState } from "react"

export function FeatureImportance() {
  const { currentData } = useWebSocket()
  const [features, setFeatures] = useState([
    { name: "Beta", value: 32.7, description: "Reduces stress level" },
    { name: "Gamma", value: 31.6, description: "Reduces stress level" },
    { name: "Theta", value: 18.9, description: "Contributes to current stress level" },
    { name: "Alpha", value: 12.6, description: "Contributes to current stress level" },
    { name: "Delta", value: 5.4, description: "Reduces stress level" },
  ])

  useEffect(() => {
    if (currentData && currentData.attentionWeights) {
      // Map attention weights to features
      const weights = Object.entries(currentData.attentionWeights)
        .filter(([key]) => key.includes("mean_") || key.includes("_ratio"))
        .map(([key, value]) => {
          const name = key.replace("mean_", "").replace("_ratio", " ratio")
          const band = name.split("_")[0]
          return {
            name: band.charAt(0).toUpperCase() + band.slice(1),
            value: Math.round(value * 1000) / 10,
            description: value > 0.2 ? "Contributes to current stress level" : "Reduces stress level",
          }
        })
        .sort((a, b) => b.value - a.value)
        .slice(0, 5)

      if (weights.length > 0) {
        setFeatures(weights)
      }
    }
  }, [currentData])

  return (
    <div className="flex flex-col">
      <h2 className="text-xl font-semibold mb-4">Feature Importance</h2>
      <div className="space-y-6">
        {features.map((feature) => (
          <div key={feature.name} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-medium" style={{ color: getBrainwaveColor(feature.name) }}>
                {feature.name}
              </span>
              <span className="text-white">{feature.value}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${feature.value}%`,
                  backgroundColor: getBrainwaveColor(feature.name),
                }}
              ></div>
            </div>
            <p className="text-xs text-white/50">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
