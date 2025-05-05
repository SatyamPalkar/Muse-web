"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { useEffect, useState } from "react"

export function EEGSignalVisualization() {
  const { currentData } = useWebSocket()
  const [chartData, setChartData] = useState<
    Array<{ index: number; alpha: number; beta: number; theta: number; delta: number; gamma: number }>
  >([])

  useEffect(() => {
    if (currentData) {
      // Get the last 50 points from each wave type
      const dataPoints = 50
      const newData = Array.from({ length: dataPoints }, (_, i) => {
        const index = i
        const alphaIdx = currentData.brainwaves.alpha.length - dataPoints + i
        const alpha = alphaIdx >= 0 ? currentData.brainwaves.alpha[alphaIdx] : 0

        const betaIdx = currentData.brainwaves.beta.length - dataPoints + i
        const beta = betaIdx >= 0 ? currentData.brainwaves.beta[betaIdx] : 0

        const thetaIdx = currentData.brainwaves.theta.length - dataPoints + i
        const theta = thetaIdx >= 0 ? currentData.brainwaves.theta[thetaIdx] : 0

        const deltaIdx = currentData.brainwaves.delta.length - dataPoints + i
        const delta = deltaIdx >= 0 ? currentData.brainwaves.delta[deltaIdx] : 0

        const gammaIdx = currentData.brainwaves.gamma.length - dataPoints + i
        const gamma = gammaIdx >= 0 ? currentData.brainwaves.gamma[gammaIdx] : 0

        return { index, alpha, beta, theta, delta, gamma }
      })

      setChartData(newData)
    }
  }, [currentData])

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-3">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">EEG Signal Visualization</CardTitle>
      </CardHeader>
      <CardContent className="p-0 pt-2">
        <ChartContainer
          config={{
            alpha: {
              label: "Alpha",
              color: "hsl(210, 100%, 60%)",
            },
            beta: {
              label: "Beta",
              color: "hsl(0, 100%, 60%)",
            },
            theta: {
              label: "Theta",
              color: "hsl(270, 100%, 60%)",
            },
            delta: {
              label: "Delta",
              color: "hsl(120, 100%, 60%)",
            },
            gamma: {
              label: "Gamma",
              color: "hsl(30, 100%, 60%)",
            },
          }}
          className="h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <XAxis dataKey="index" tick={false} />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line
                type="monotone"
                dataKey="alpha"
                stroke="var(--color-alpha)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="beta"
                stroke="var(--color-beta)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="theta"
                stroke="var(--color-theta)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="delta"
                stroke="var(--color-delta)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="gamma"
                stroke="var(--color-gamma)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
