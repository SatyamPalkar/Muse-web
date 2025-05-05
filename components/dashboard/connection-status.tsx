"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { cn } from "@/lib/utils"
import { AlertCircle, CheckCircle2 } from "lucide-react"

export function ConnectionStatus() {
  const { connected } = useWebSocket()

  return (
    <div className="flex items-center gap-2 rounded-full border border-white/10 bg-gradient-to-r from-slate-800/50 to-slate-900/50 backdrop-blur-sm px-3 py-1.5 text-sm shadow-lg">
      <div
        className={cn(
          "h-2.5 w-2.5 rounded-full animate-pulse",
          connected ? "bg-emerald-400 shadow-md shadow-emerald-500/50" : "bg-rose-400 shadow-md shadow-rose-500/50",
        )}
      />
      <span>
        {connected ? (
          <span className="flex items-center gap-1 font-medium text-emerald-300">
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            Connected
          </span>
        ) : (
          <span className="flex items-center gap-1 font-medium text-rose-300">
            <AlertCircle className="h-4 w-4 text-rose-400" />
            Disconnected
          </span>
        )}
      </span>
    </div>
  )
}
