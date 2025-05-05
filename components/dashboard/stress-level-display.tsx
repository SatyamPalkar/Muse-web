"use client"

import { useEffect, useState } from "react"
import { useWebSocket } from "@/contexts/websocket-context"
import { getStressLevelFromClass, getStressLevelColor, getStressLevelDescription } from "@/utils/stress-level"
import type { StressLevel } from "@/types/eeg"
import { Activity } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function StressLevelDisplay() {
  const { currentData } = useWebSocket()
  const [currentLevel, setCurrentLevel] = useState<StressLevel>("Positive")
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    if (currentData) {
      const newLevel = getStressLevelFromClass(currentData.classification)
      if (newLevel !== currentLevel) {
        setIsTransitioning(true)
        const timer = setTimeout(() => {
          setCurrentLevel(newLevel)
          setIsTransitioning(false)
        }, 300)
        return () => clearTimeout(timer)
      }
    }
  }, [currentData, currentLevel])

  const levelColor = getStressLevelColor(currentLevel)
  const description = getStressLevelDescription(currentLevel)

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Current Stress Level</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center p-6">
        <div
          className={`relative flex h-48 w-48 items-center justify-center rounded-full ${levelColor} transition-colors duration-500 ${isTransitioning ? "scale-95 opacity-90" : "scale-100 opacity-100"}`}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <Activity className="h-24 w-24 text-white" />
          </div>
          <div className="absolute inset-0 rounded-full bg-white/10 animate-pulse-opacity" />
        </div>
        <h2
          className={`mt-6 text-4xl font-bold transition-all duration-500 ${isTransitioning ? "opacity-0 -translate-y-2" : "opacity-100 translate-y-0"}`}
        >
          {currentLevel}
        </h2>
        <p className="mt-2 text-center text-muted-foreground">{description}</p>
        <p className="mt-4 text-sm text-muted-foreground">
          {currentData ? new Date(currentData.timestamp).toLocaleTimeString() : "Waiting for data..."}
        </p>
      </CardContent>
    </Card>
  )
}
