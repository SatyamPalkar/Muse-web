"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { useEffect, useRef } from "react"
import { ArrowUpRight, Clock } from "lucide-react"

export function HistoricalTrend() {
  const { historicalData } = useWebSocket()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth * window.devicePixelRatio
    canvas.height = canvas.offsetHeight * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // If no data, return
    if (historicalData.length === 0) return

    // Map states to y-positions (0 = Relaxed, 1 = Stressed, 2 = Focused, 3 = Drowsy)
    const stateToY = (state: string) => {
      switch (state) {
        case "Relaxed":
          return 3
        case "Stressed":
          return 1
        case "Focused":
          return 2
        case "Drowsy":
          return 0
        default:
          return 3
      }
    }

    // Draw the line
    ctx.beginPath()
    ctx.strokeStyle = "#3a85ff" // Blue line
    ctx.lineWidth = 2
    ctx.lineJoin = "round"
    ctx.lineCap = "round"

    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const stepX = width / (historicalData.length - 1 || 1)
    const stepY = height / 4

    historicalData.forEach((point, i) => {
      const x = i * stepX
      const y = stateToY(point.state) * stepY + stepY / 2

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()

    // Add glow effect
    ctx.shadowColor = "#3a85ff"
    ctx.shadowBlur = 5
    ctx.stroke()
  }, [historicalData])

  // Create time labels for x-axis
  const timeLabels = []
  if (historicalData.length > 0) {
    const interval = Math.floor(historicalData.length / 5)
    for (let i = 0; i < 5; i++) {
      const index = i * interval
      if (index < historicalData.length) {
        const time = new Date(historicalData[index].timestamp).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
        timeLabels.push(time)
      }
    }
  }

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Historical Trend</h2>
        <div className="flex gap-2">
          <button className="flex items-center gap-1 bg-white/10 px-3 py-1 rounded-full text-sm">
            <ArrowUpRight className="h-4 w-4" />
            Stress
          </button>
          <button className="flex items-center gap-1 bg-transparent border border-white/20 px-3 py-1 rounded-full text-sm">
            <Clock className="h-4 w-4" />
            Brainwaves
          </button>
        </div>
      </div>

      <div className="relative h-[200px] w-full">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full"></canvas>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-white/50 py-2">
          <span>Drowsy</span>
          <span>Stressed</span>
          <span>Focused</span>
          <span>Relaxed</span>
        </div>

        {/* X-axis labels */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-white/50 px-10">
          {timeLabels.map((time, i) => (
            <span key={i}>{time}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
