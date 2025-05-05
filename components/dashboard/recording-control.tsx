"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { Button } from "@/components/ui/button"
import { Play, Square } from "lucide-react"

export function RecordingControl() {
  const { isRecording, startRecording, stopRecording } = useWebSocket()

  return (
    <div className="flex items-center gap-2">
      {isRecording ? (
        <Button
          variant="destructive"
          size="sm"
          onClick={stopRecording}
          className="flex items-center gap-1 bg-gradient-to-r from-rose-600 to-pink-600 hover:from-rose-500 hover:to-pink-500 border-none shadow-lg hover:shadow-rose-500/30 transition-all duration-300"
        >
          <Square className="h-4 w-4" />
          Stop Recording
        </Button>
      ) : (
        <Button
          variant="default"
          size="sm"
          onClick={startRecording}
          className="flex items-center gap-1 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 border-none shadow-lg hover:shadow-emerald-500/30 transition-all duration-300"
        >
          <Play className="h-4 w-4" />
          Start Recording
        </Button>
      )}
    </div>
  )
}
