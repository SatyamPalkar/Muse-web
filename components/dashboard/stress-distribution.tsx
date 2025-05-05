"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getStressLevelColor, getStressLevelTextColor } from "@/utils/stress-level"
import { Progress } from "@/components/ui/progress"

export function StressDistribution() {
  const { stressDistribution } = useWebSocket()

  // Format time in minutes and seconds
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  return (
    <Card className="col-span-1 md:col-span-1 lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Stress Distribution</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {stressDistribution.map((item) => (
          <div key={item.level} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className={`font-medium ${getStressLevelTextColor(item.level)}`}>{item.level}</span>
              <span className="text-sm text-muted-foreground">{formatTime(item.timeSpent)}</span>
            </div>
            <Progress value={item.percentage} className={`h-2 ${getStressLevelColor(item.level)} bg-muted/30`} />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
