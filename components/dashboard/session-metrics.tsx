"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getStressLevelTextColor } from "@/utils/stress-level"
import { Clock, Activity, BarChart, ArrowUpDown } from "lucide-react"

export function SessionMetrics() {
  const { sessionMetrics, isRecording } = useWebSocket()

  // Format duration in minutes and seconds
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  // Format average stress level
  const formatAvgStress = (level: number) => {
    if (level < 0.5) return "Low"
    if (level < 1.5) return "Moderate"
    if (level < 2.5) return "High"
    return "Severe"
  }

  if (isRecording) {
    return (
      <Card className="col-span-1 md:col-span-1 lg:col-span-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl">Session Metrics</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center p-6">
          <div className="flex h-24 w-24 items-center justify-center rounded-full bg-primary/10 animate-pulse">
            <Activity className="h-12 w-12 text-primary" />
          </div>
          <h3 className="mt-4 text-xl font-semibold">Recording in progress</h3>
          <p className="text-sm text-muted-foreground">Session metrics will be available when recording stops</p>
        </CardContent>
      </Card>
    )
  }

  if (!sessionMetrics) {
    return (
      <Card className="col-span-1 md:col-span-1 lg:col-span-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl">Session Metrics</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center p-6">
          <div className="flex h-24 w-24 items-center justify-center rounded-full bg-muted/30">
            <Activity className="h-12 w-12 text-muted-foreground" />
          </div>
          <h3 className="mt-4 text-xl font-semibold">No session data</h3>
          <p className="text-sm text-muted-foreground">Start a recording to collect session metrics</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="col-span-1 md:col-span-1 lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Session Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col space-y-1 rounded-lg border p-3">
            <div className="flex items-center text-sm text-muted-foreground">
              <Clock className="mr-1 h-4 w-4" />
              Duration
            </div>
            <div className="text-lg font-semibold">{formatDuration(sessionMetrics.duration)}</div>
          </div>

          <div className="flex flex-col space-y-1 rounded-lg border p-3">
            <div className="flex items-center text-sm text-muted-foreground">
              <BarChart className="mr-1 h-4 w-4" />
              Avg. Stress
            </div>
            <div className="text-lg font-semibold">{formatAvgStress(sessionMetrics.averageStressLevel)}</div>
          </div>

          <div className="flex flex-col space-y-1 rounded-lg border p-3">
            <div className="flex items-center text-sm text-muted-foreground">
              <Activity className="mr-1 h-4 w-4" />
              Peak Level
            </div>
            <div className={`text-lg font-semibold ${getStressLevelTextColor(sessionMetrics.peakStressLevel)}`}>
              {sessionMetrics.peakStressLevel}
            </div>
          </div>

          <div className="flex flex-col space-y-1 rounded-lg border p-3">
            <div className="flex items-center text-sm text-muted-foreground">
              <ArrowUpDown className="mr-1 h-4 w-4" />
              Transitions
            </div>
            <div className="text-lg font-semibold">{sessionMetrics.stressTransitions}</div>
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          Session started: {new Date(sessionMetrics.startTime).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  )
}
