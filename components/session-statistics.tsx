"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { getBrainStateTextColor } from "@/utils/brain-state"
import { Clock, Activity, Zap } from "lucide-react"

export function SessionStatistics() {
  const { sessionMetrics } = useWebSocket()

  // Format duration in minutes and seconds
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  return (
    <div className="flex flex-col">
      <h2 className="text-xl font-semibold mb-4">Session Statistics</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="card-border p-4">
          <div className="flex items-center gap-2 mb-2 text-white/50">
            <Clock className="h-4 w-4" />
            <span className="text-sm">Session Duration</span>
          </div>
          <div className="text-2xl font-bold">{sessionMetrics ? formatDuration(sessionMetrics.duration) : "0m 0s"}</div>
        </div>

        <div className="card-border p-4">
          <div className="flex items-center gap-2 mb-2 text-white/50">
            <Activity className="h-4 w-4" />
            <span className="text-sm">Calm Periods</span>
          </div>
          <div className="text-2xl font-bold">{sessionMetrics ? sessionMetrics.calmPeriods : 0}</div>
        </div>

        <div className="card-border p-4">
          <div className="flex items-center gap-2 mb-2 text-white/50">
            <Zap className="h-4 w-4" />
            <span className="text-sm">Peak Stress</span>
          </div>
          <div
            className={`text-2xl font-bold ${
              sessionMetrics ? getBrainStateTextColor(sessionMetrics.peakState) : "text-white"
            }`}
          >
            {sessionMetrics ? sessionMetrics.peakState : "None"}
          </div>
        </div>
      </div>
    </div>
  )
}
