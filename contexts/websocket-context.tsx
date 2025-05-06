"use client"

import React, { createContext, useContext, useEffect, useState } from "react"

interface WebSocketContextType {
  ws: WebSocket | null
  emotion: string | null
}

const WebSocketContext = createContext<WebSocketContextType>({
  ws: null,
  emotion: null,
})

export const useWebSocket = () => useContext(WebSocketContext)

export const WebSocketProvider = ({
  children,
  socketUrl = "wss://muse-eeg-backend.onrender.com", // your backend WebSocket endpoint
}: {
  children: React.ReactNode
  socketUrl?: string
}) => {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [emotion, setEmotion] = useState<string | null>(null)

  useEffect(() => {
    const socket = new WebSocket(socketUrl)

    socket.onopen = () => {
      console.log("✅ WebSocket connected")
    }

    socket.onmessage = (event) => {
      console.log("🎯 Emotion:", event.data)
      setEmotion(event.data)
    }

    socket.onerror = (error) => {
      console.error("❌ WebSocket error:", error)
    }

    socket.onclose = () => {
      console.log("❗ WebSocket disconnected")
    }

    setWs(socket)
    return () => {
      socket.close()
    }
  }, [socketUrl])

  return (
    <WebSocketContext.Provider value={{ ws, emotion }}>
      {children}
    </WebSocketContext.Provider>
  )
}
