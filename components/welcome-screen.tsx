"use client"

import { Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useWebSocket } from "@/contexts/websocket-context"
import { useRouter } from "next/navigation"

export function WelcomeScreen() {
  const { startRecording } = useWebSocket()
  const router = useRouter()

  const handleStart = () => {
    startRecording()
    router.push("/dashboard")
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="flex items-center gap-3 mb-8">
        <Activity className="h-8 w-8 text-white" />
        <h1 className="text-3xl font-bold">EEG Stress Monitor</h1>
      </div>

      <p className="text-center text-white/70 max-w-md mb-12">
        Connect to the EEG device to start monitoring brain activity and stress levels in real-time.
      </p>

      <Button
        onClick={handleStart}
        className="bg-white text-black hover:bg-white/90 px-6 py-6 rounded-md flex items-center gap-2"
      >
        <span className="i-lucide-play h-5 w-5" />
        Start Monitoring
      </Button>
    </div>
  )
}
