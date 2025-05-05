"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { StressLevel } from "@/types/eeg"

export function HistoricalTrend() {
  const { historicalData } = useWebSocket()

  // Convert stress levels to numerical values for the chart
  const chartData = historicalData.map((item, index) => {
    const levelValue =
      item.level === "Positive"
        ? 0
        : item.level === "Acute"
          ? 1
          : item.level === "Episodic"
            ? 2
            : item.level === "Toxic"
              ? 3
              : 0

    return {
      index,
      timestamp: new Date(item.timestamp).toLocaleTimeString(),
      level: levelValue,
      levelName: item.level,
    }
  })

  // Custom tooltip to show the actual level name
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="rounded-lg border bg-background p-2 shadow-sm">
          <p className="text-sm">{data.timestamp}</p>
          <p className="font-bold">{data.levelName}</p>
        </div>
      )
    }
    return null
  }

  // Custom YAxis tick to show level names instead of numbers
  const CustomYAxisTick = ({ x, y, payload }: any) => {
    if (!payload || payload.value === undefined) {
      return null
    }

    const levelMap: Record<number, StressLevel> = {
      0: "Positive",
      1: "Acute",
      2: "Episodic",
      3: "Toxic",
    }
    const level = levelMap[payload.value] || ""
    return (
      <text x={x} y={y} dy={4} fontSize={12} textAnchor="end" fill="currentColor">
        {level}
      </text>
    )
  }

  return (
    <Card className="col-span-1 md:col-span-3 lg:col-span-3">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Stress Level History</CardTitle>
      </CardHeader>
      <CardContent className="p-0 pt-2">
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 60, bottom: 0 }}>
              <defs>
                <linearGradient id="colorLevel" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.2} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="index"
                tick={false}
                label={{ value: "Time", position: "insideBottomRight", offset: -10 }}
              />
              <YAxis dataKey="level" domain={[0, 3]} ticks={[0, 1, 2, 3]} tick={<CustomYAxisTick />} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="stepAfter"
                dataKey="level"
                stroke="#f43f5e"
                fillOpacity={1}
                fill="url(#colorLevel)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
