"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Brain, 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Zap,
  Heart,
  Target,
  BarChart3,
  Home,
  Settings
} from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"

// Mock data for demonstration
const mockEEGData = [
  { time: "14:30:00", theta: 15, alpha: 25, beta: 18, gamma: 8, delta: 20, stress: 0.2 },
  { time: "14:30:05", theta: 18, alpha: 28, beta: 20, gamma: 10, delta: 22, stress: 0.3 },
  { time: "14:30:10", theta: 12, alpha: 22, beta: 25, gamma: 12, delta: 18, stress: 0.5 },
  { time: "14:30:15", theta: 10, alpha: 20, beta: 35, gamma: 15, delta: 15, stress: 0.7 },
  { time: "14:30:20", theta: 14, alpha: 24, beta: 22, gamma: 9, delta: 19, stress: 0.4 },
]

interface StressMetrics {
  level: string
  confidence: number
  trend: "up" | "down" | "stable"
  betaAlpha: number
  recommendations: string[]
}

export default function DashboardPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [currentMetrics, setCurrentMetrics] = useState<StressMetrics>({
    level: "Moderate Stress",
    confidence: 0.85,
    trend: "stable",
    betaAlpha: 1.2,
    recommendations: [
      "Consider taking a short break",
      "Practice deep breathing exercises",
      "Adjust your environment lighting"
    ]
  })

  const [sessionStats, setSessionStats] = useState({
    duration: "05:42",
    samples: 342,
    avgStress: 0.42,
    peakStress: 0.78
  })

  const getStressColor = (level: string) => {
    if (level.includes("Very High") || level.includes("High")) return "bg-red-500"
    if (level.includes("Moderate")) return "bg-yellow-500"
    if (level.includes("Light")) return "bg-orange-500"
    return "bg-green-500"
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up": return <TrendingUp className="h-4 w-4 text-red-500" />
      case "down": return <TrendingDown className="h-4 w-4 text-green-500" />
      default: return <Activity className="h-4 w-4 text-blue-500" />
    }
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => window.location.href = '/'}>
              <Home className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-3xl font-bold">EEG Dashboard</h1>
              <p className="text-muted-foreground">Real-time brain activity monitoring</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={isConnected ? "default" : "destructive"}>
              {isConnected ? "Connected" : "Disconnected"}
            </Badge>
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Stress Level</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{currentMetrics.level}</div>
              <div className="flex items-center gap-2 mt-2">
                <div className={`w-3 h-3 rounded-full ${getStressColor(currentMetrics.level)}`} />
                <span className="text-sm text-muted-foreground">
                  {(currentMetrics.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Trend</CardTitle>
              {getTrendIcon(currentMetrics.trend)}
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold capitalize">{currentMetrics.trend}</div>
              <p className="text-xs text-muted-foreground">
                Beta/Alpha: {currentMetrics.betaAlpha}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Session Time</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{sessionStats.duration}</div>
              <p className="text-xs text-muted-foreground">
                {sessionStats.samples} samples collected
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Stress</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{sessionStats.avgStress.toFixed(2)}</div>
              <Progress value={sessionStats.avgStress * 100} className="mt-2" />
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard */}
        <Tabs defaultValue="realtime" className="space-y-4">
          <TabsList>
            <TabsTrigger value="realtime">Real-time</TabsTrigger>
            <TabsTrigger value="session">Session Analysis</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="realtime" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              
              {/* EEG Brainwave Chart */}
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>EEG Brainwaves</CardTitle>
                  <CardDescription>Real-time frequency band powers</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={mockEEGData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="theta" stroke="#8884d8" strokeWidth={2} name="Theta" />
                      <Line type="monotone" dataKey="alpha" stroke="#82ca9d" strokeWidth={2} name="Alpha" />
                      <Line type="monotone" dataKey="beta" stroke="#ffc658" strokeWidth={2} name="Beta" />
                      <Line type="monotone" dataKey="gamma" stroke="#ff7300" strokeWidth={2} name="Gamma" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Stress Timeline */}
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>Stress Timeline</CardTitle>
                  <CardDescription>Stress level over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={mockEEGData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="stress" 
                        stroke="#ef4444" 
                        fill="#ef444420"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle>Current Recommendations</CardTitle>
                <CardDescription>AI-powered suggestions based on your brain activity</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {currentMetrics.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg">
                      <Zap className="h-4 w-4 text-blue-500" />
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="session" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>30-Second Analysis</CardTitle>
                  <CardDescription>Short-term assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">Light Stress</div>
                    <div className="text-sm text-muted-foreground">Confidence: 76%</div>
                    <Progress value={76} className="mt-2" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>60-Second Analysis</CardTitle>
                  <CardDescription>Medium-term assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">Moderate Stress</div>
                    <div className="text-sm text-muted-foreground">Confidence: 84%</div>
                    <Progress value={84} className="mt-2" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>90-Second Analysis</CardTitle>
                  <CardDescription>Extended assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">Moderate Stress</div>
                    <div className="text-sm text-muted-foreground">Confidence: 89%</div>
                    <Progress value={89} className="mt-2" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>120-Second Analysis</CardTitle>
                  <CardDescription>Comprehensive assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">Mixed State</div>
                    <div className="text-sm text-muted-foreground">Confidence: 92%</div>
                    <Progress value={92} className="mt-2" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Session History</CardTitle>
                <CardDescription>Previous analysis sessions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No previous sessions found</p>
                  <p className="text-sm">Start a session to see historical data</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
