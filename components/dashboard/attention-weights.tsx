"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { useState, useEffect } from "react"

export function AttentionWeights() {
  const { currentData } = useWebSocket()
  const [chartData, setChartData] = useState<Array<{ feature: string; weight: number }>>([])

  useEffect(() => {
    if (currentData && currentData.attentionWeights) {
      const newData = Object.entries(currentData.attentionWeights).map(([feature, weight]) => ({
        feature,
        weight: Number.parseFloat(weight.toFixed(3)),
      }))

      // Sort by weight descending
      newData.sort((a, b) => b.weight - a.weight)

      // Take top 10 features
      setChartData(newData.slice(0, 10))
    }
  }, [currentData])

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-3">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Attention Weights</CardTitle>
      </CardHeader>
      <CardContent className="p-0 pt-2">
        <ChartContainer
          config={{
            weight: {
              label: "Weight",
              color: "hsl(340, 70%, 60%)",
            },
          }}
          className="h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 20, right: 30, left: 60, bottom: 0 }}>
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={100} tick={{ fontSize: 12 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="weight" fill="var(--color-weight)" radius={[0, 4, 4, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
