"use client"

import { Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useWebSocket } from "@/contexts/websocket-context"
import { Moon, Sun } from "lucide-react"
import { useState } from "react"

export function Header() {
  const { connected, isRecording, startRecording, stopRecording } = useWebSocket()
  const [isDarkMode, setIsDarkMode] = useState(true)

  return (
    <header className="flex h-16 items-center justify-between px-6 border-b border-white/10">
      <div className="flex items-center gap-2">
        <Activity className="h-6 w-6 text-white" />
        <h1 className="text-xl font-bold text-white">EEG Stress Monitor</h1>
      </div>
      <div className="flex items-center gap-4">
        <Button
          variant="outline"
          size="sm"
          className={`border border-white/20 bg-transparent hover:bg-white/5 ${
            connected ? "text-green-400" : "text-red-400"
          }`}
          onClick={connected ? stopRecording : startRecording}
        >
          {connected ? "Disconnect" : "Connect"}
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="border border-white/20 bg-transparent hover:bg-white/5"
          onClick={() => setIsDarkMode(!isDarkMode)}
        >
          {isDarkMode ? <Sun className="h-4 w-4 text-white" /> : <Moon className="h-4 w-4 text-white" />}
        </Button>
      </div>
    </header>
  )
}
