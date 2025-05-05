"use client"

import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"
import type { Socket } from "socket.io-client"
import type {
  EEGData,
  HistoricalData,
  StateDistribution,
  BrainWaveDistribution,
  SessionMetrics,
  BrainState,
} from "@/types/eeg"
import { getBrainStateFromClass } from "@/utils/brain-state"

interface WebSocketContextType {
  socket: Socket | null
  connected: boolean
  currentData: EEGData | null
  historicalData: HistoricalData[]
  stateDistribution: StateDistribution[]
  brainwaveDistribution: BrainWaveDistribution[]
  sessionMetrics: SessionMetrics | null
  isRecording: boolean
  startRecording: () => void
  stopRecording: () => void
}

const WebSocketContext = createContext<WebSocketContextType>({
  socket: null,
  connected: false,
  currentData: null,
  historicalData: [],
  stateDistribution: [],
  brainwaveDistribution: [],
  sessionMetrics: null,
  isRecording: false,
  startRecording: () => {},
  stopRecording: () => {},
})

export const useWebSocket = () => useContext(WebSocketContext)

export const WebSocketProvider = ({
  children,
  socketUrl = "ws://localhost:3001",
}: {
  children: React.ReactNode
  socketUrl?: string
}) => {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [connected, setConnected] = useState(false)
  const [currentData, setCurrentData] = useState<EEGData | null>(null)
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([])
  const [stateDistribution, setStateDistribution] = useState<StateDistribution[]>([
    { state: "Relaxed", timeSpent: 0, percentage: 0 },
    { state: "Stressed", timeSpent: 0, percentage: 0 },
    { timeSpent: 0, percentage: 0 },
    { state: "Stressed", timeSpent: 0, percentage: 0 },
    { state: "Focused", timeSpent: 0, percentage: 0 },
    { state: "Drowsy", timeSpent: 0, percentage: 0 },
  ])
  const [brainwaveDistribution, setBrainwaveDistribution] = useState<BrainWaveDistribution[]>([
    { band: "Alpha", power: 0, description: "Contributes to current stress level" },
    { band: "Beta", power: 0, description: "Reduces stress level" },
    { band: "Theta", power: 0, description: "Contributes to current stress level" },
    { band: "Delta", power: 0, description: "Contributes to current stress level" },
    { band: "Gamma", power: 0, description: "Reduces stress level" },
  ])
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetrics | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null)
  const [calmPeriods, setCalmPeriods] = useState(0)
  const [lastState, setLastState] = useState<BrainState | null>(null)
  const [peakState, setPeakState] = useState<BrainState>("Relaxed")

  // Mock data generation for demo purposes
  useEffect(() => {
    if (isRecording && !connected) {
      const interval = setInterval(() => {
        const timestamp = Date.now()
        const classification = Math.floor(Math.random() * 4)
        const confidence = Math.floor(Math.random() * 30) + 70
        const currentState = getBrainStateFromClass(classification)

        // Generate mock EEG data
        const mockData: EEGData = {
          timestamp,
          rawValues: Array(100)
            .fill(0)
            .map(() => Math.random() * 100),
          brainwaves: {
            alpha: Array(50)
              .fill(0)
              .map(() => Math.random() * 100),
            beta: Array(50)
              .fill(0)
              .map(() => Math.random() * 100),
            theta: Array(50)
              .fill(0)
              .map(() => Math.random() * 100),
            delta: Array(50)
              .fill(0)
              .map(() => Math.random() * 100),
            gamma: Array(50)
              .fill(0)
              .map(() => Math.random() * 100),
          },
          features: {
            mean_alpha: Math.random() * 50,
            mean_beta: Math.random() * 50,
            mean_theta: Math.random() * 50,
            mean_delta: Math.random() * 50,
            mean_gamma: Math.random() * 50,
          },
          classification,
          confidence,
          attentionWeights: {
            mean_alpha: Math.random(),
            mean_beta: Math.random(),
            mean_theta: Math.random(),
            mean_delta: Math.random(),
            mean_gamma: Math.random(),
          },
        }

        // Normalize attention weights
        const sum = Object.values(mockData.attentionWeights).reduce((a, b) => a + b, 0)
        Object.keys(mockData.attentionWeights).forEach((key) => {
          mockData.attentionWeights[key] = mockData.attentionWeights[key] / sum
        })

        setCurrentData(mockData)

        // Update historical data
        setHistoricalData((prev) => {
          const newData = [
            ...prev,
            {
              timestamp,
              state: currentState,
            },
          ]
          return newData.slice(-100)
        })

        // Update state distribution
        setStateDistribution((prev) => {
          const updated = prev.map((item) => {
            if (item.state === currentState) {
              return { ...item, timeSpent: item.timeSpent + 1 }
            }
            return item
          })

          const totalTime = updated.reduce((sum, item) => sum + item.timeSpent, 0)
          return updated.map((item) => ({
            ...item,
            percentage: totalTime > 0 ? (item.timeSpent / totalTime) * 100 : 0,
          }))
        })

        // Update calm periods
        if (lastState === "Stressed" && currentState === "Relaxed") {
          setCalmPeriods((prev) => prev + 1)
        }

        // Update peak state
        const stateValues: Record<BrainState, number> = {
          Relaxed: 0,
          Focused: 1,
          Drowsy: 2,
          Stressed: 3,
        }

        if (stateValues[currentState] > stateValues[peakState]) {
          setPeakState(currentState)
        }

        setLastState(currentState)
      }, 1000)

      return () => clearInterval(interval)
    }
  }, [isRecording, connected, lastState, peakState])

  const startRecording = () => {
    setIsRecording(true)
    setConnected(true)
    setSessionStartTime(Date.now())
    setHistoricalData([])
    setStateDistribution([
      { state: "Relaxed", timeSpent: 0, percentage: 0 },
      { state: "Stressed", timeSpent: 0, percentage: 0 },
      { state: "Focused", timeSpent: 0, percentage: 0 },
      { state: "Drowsy", timeSpent: 0, percentage: 0 },
    ])
    setCalmPeriods(0)
    setLastState(null)
    setPeakState("Relaxed")
  }

  const stopRecording = () => {
    if (isRecording && sessionStartTime) {
      setIsRecording(false)
      setConnected(false)

      setSessionMetrics({
        startTime: sessionStartTime,
        duration: Date.now() - sessionStartTime,
        calmPeriods,
        peakState,
      })
    }
  }

  useEffect(() => {
    // In a real app, we would connect to the WebSocket server here
    // For demo purposes, we're using mock data
    return () => {
      if (socket) {
        socket.disconnect()
      }
    }
  }, [socket])

  return (
    <WebSocketContext.Provider
      value={{
        socket,
        connected,
        currentData,
        historicalData,
        stateDistribution,
        brainwaveDistribution,
        sessionMetrics,
        isRecording,
        startRecording,
        stopRecording,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  )
}
