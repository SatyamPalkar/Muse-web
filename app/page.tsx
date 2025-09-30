"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, Activity, TrendingUp, Settings } from "lucide-react"
import { ModeToggle } from "@/components/mode-toggle"

export default function HomePage() {
  const [isConnected, setIsConnected] = useState(false)
  const [currentStress, setCurrentStress] = useState("Unknown")

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-600 rounded-lg">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Muse EEG Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                Real-time stress detection and analysis
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <ModeToggle />
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Status Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Connection Status</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {isConnected ? "Connected" : "Disconnected"}
              </div>
              <div className={`w-3 h-3 rounded-full mt-2 ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Current Stress Level</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{currentStress}</div>
              <p className="text-xs text-muted-foreground">
                Based on real-time EEG analysis
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Session Duration</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">00:00</div>
              <p className="text-xs text-muted-foreground">
                Active monitoring time
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-6">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Getting Started</CardTitle>
              <CardDescription>
                Connect your Muse headband and start monitoring your brain activity
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col space-y-2">
                <h3 className="font-semibold">Step 1: Start the Backend</h3>
                <code className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm">
                  python master_eeg_analyzer.py --mode realtime
                </code>
              </div>
              
              <div className="flex flex-col space-y-2">
                <h3 className="font-semibold">Step 2: Configure Mind Monitor</h3>
                <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
                  <li>• Set IP to your computer's IP address</li>
                  <li>• Set Port to 8000</li>
                  <li>• Enable EEG streaming</li>
                </ul>
              </div>

              <div className="flex flex-col space-y-2">
                <h3 className="font-semibold">Step 3: Monitor Your Data</h3>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Visit the <Button variant="link" className="p-0 h-auto">dashboard</Button> to see real-time analysis
                </p>
              </div>

              <div className="pt-4">
                <Button 
                  onClick={() => window.location.href = '/dashboard'}
                  className="w-full sm:w-auto"
                >
                  Go to Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
