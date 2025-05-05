"use client"

import { useEffect, useState } from "react"
import { useWebSocket } from "@/contexts/websocket-context"
import { getBrainStateFromClass, getBrainStateColor } from "@/utils/brain-state"
import type { BrainState } from "@/types/eeg"
import { Brain } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function BrainStateDisplay() {
  const { currentData } = useWebSocket()
  const [currentState, setCurrentState] = useState<BrainState>("Relaxed")
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    if (currentData) {
      const newState = getBrainStateFromClass(currentData.classification)
      if (newState !== currentState) {
        setIsTransitioning(true)
        const timer = setTimeout(() => {
          setCurrentState(newState)
          setIsTransitioning(false)
        }, 300)
        return () => clearTimeout(timer)
      }
    }
  }, [currentData, currentState])

  const stateColor = getBrainStateColor(currentState)

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Current Brain State</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center p-6">
        <div
          className={`relative flex h-48 w-48 items-center justify-center rounded-full ${stateColor} transition-colors duration-500 ${isTransitioning ? "scale-95 opacity-90" : "scale-100 opacity-100"}`}
        >
          <Brain className="h-24 w-24 text-white" />
          <div className="absolute inset-0 rounded-full bg-white/10 animate-pulse-opacity" />
        </div>
        <h2
          className={`mt-6 text-4xl font-bold transition-all duration-500 ${isTransitioning ? "opacity-0 -translate-y-2" : "opacity-100 translate-y-0"}`}
        >
          {currentState}
        </h2>
        <p className="mt-2 text-muted-foreground">
          {currentData ? new Date(currentData.timestamp).toLocaleTimeString() : "Waiting for data..."}
        </p>
      </CardContent>
    </Card>
  )
}
