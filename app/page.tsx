"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { WelcomeScreen } from "@/components/welcome-screen"
import { useEffect } from "react"

export default function Home() {
  const { currentData, connected } = useWebSocket()

  useEffect(() => {
    if (connected && currentData) {
      console.log("📡 EEG Data Received:", currentData)
    }
  }, [currentData, connected])

  return <WelcomeScreen />
}
