
"use client"

import React, { createContext, useContext, useEffect, useState } from "react"

interface EEGData {
  emotion: string
  timestamp: number
}

interface WebSocketContextType {
  connected: boolean
  currentData: EEGData | null
}

const WebSocketContext = createContext<WebSocketContextType>({
  connected: false,
  currentData: null,
})

export const useWebSocket = () => useContext(WebSocketContext)

export const WebSocketProvider = ({
  children,
}: {
  children: React.ReactNode
}) => {
  const [connected, setConnected] = useState(false)
  const [currentData, setCurrentData] = useState<EEGData | null>(null)

  useEffect(() => {
    const socket = new WebSocket("wss://muse-eeg-backend.onrender.com")

    socket.onopen = () => {
      console.log("✅ Connected to EEG WebSocket")
      setConnected(true)
    }

    socket.onmessage = (event) => {
      const emotion = event.data
      setCurrentData({
        emotion,
        timestamp: Date.now(),
      })
    }

    socket.onclose = () => {
      console.log("❌ WebSocket disconnected")
      setConnected(false)
    }

    return () => {
      socket.close()
    }
  }, [])

  return (
    <WebSocketContext.Provider value={{ connected, currentData }}>
      {children}
    </WebSocketContext.Provider>
  )
}
