"use client"

import { useState, useEffect, useCallback, useRef } from 'react'
import { EEGWebSocketClient } from '@/lib/websocket'

interface EEGRealtimeData {
  timestamp: string
  stress_level: string
  confidence: number
  overall_stress: number
  arousal_level: number
  temporal_trend: string
  recommendations: string[]
  stress_indicators: Record<string, number>
  sample_count: number
}

interface EEGSessionData {
  timestamp: string
  samples_collected: number
  elapsed_time: number
  is_collecting: boolean
}

interface EEGConnectionStatus {
  connected: boolean
  reconnecting: boolean
  error: string | null
}

export function useEEGData() {
  const [realtimeData, setRealtimeData] = useState<EEGRealtimeData | null>(null)
  const [sessionData, setSessionData] = useState<EEGSessionData | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<EEGConnectionStatus>({
    connected: false,
    reconnecting: false,
    error: null
  })
  
  const wsClient = useRef<EEGWebSocketClient | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null)
  const pollingInterval = useRef<NodeJS.Timeout | null>(null)

  // Initialize WebSocket connection for real-time updates
  const connect = useCallback(async () => {
    if (wsClient.current?.isConnected) {
      return
    }

    try {
      setConnectionStatus(prev => ({ ...prev, reconnecting: true, error: null }))
      
      wsClient.current = new EEGWebSocketClient()
      
      // Set up event listeners
      wsClient.current.on('connected', () => {
        setConnectionStatus({
          connected: true,
          reconnecting: false,
          error: null
        })
        console.log('🧠 Connected to EEG WebSocket - Real-time updates active')
      })

      wsClient.current.on('disconnected', () => {
        setConnectionStatus(prev => ({
          ...prev,
          connected: false,
          reconnecting: false
        }))
        console.log('🧠 Disconnected from EEG WebSocket')
      })

      wsClient.current.on('error', (error) => {
        setConnectionStatus(prev => ({
          ...prev,
          connected: false,
          reconnecting: false,
          error: error?.error || 'WebSocket connection error'
        }))
        console.error('🧠 EEG WebSocket error:', error)
      })

      wsClient.current.on('realtime', (data: EEGRealtimeData) => {
        setRealtimeData(data)
      })

      wsClient.current.on('session', (data: EEGSessionData) => {
        setSessionData(data)
      })

      // Connect to WebSocket
      await wsClient.current.connect()
      
    } catch (error) {
      setConnectionStatus({
        connected: false,
        reconnecting: false,
        error: error instanceof Error ? error.message : 'Connection failed'
      })
      console.error('Failed to connect to EEG WebSocket:', error)
    }
  }, [])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
      reconnectTimeout.current = null
    }
    
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current)
      pollingInterval.current = null
    }
    
    if (wsClient.current) {
      wsClient.current.disconnect()
      wsClient.current = null
    }
    
    setConnectionStatus({
      connected: false,
      reconnecting: false,
      error: null
    })
  }, [])

  // Send command to backend
  const sendCommand = useCallback((command: string, data?: any) => {
    if (wsClient.current?.isConnected) {
      wsClient.current.send(command, data)
    } else {
      console.warn('WebSocket not connected, cannot send command:', command)
    }
  }, [])

  // Auto-reconnect on mount and when connection is lost
  useEffect(() => {
    connect()

    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Auto-reconnect when connection is lost (for HTTP polling)
  useEffect(() => {
    if (!connectionStatus.connected && !connectionStatus.reconnecting && !connectionStatus.error) {
      reconnectTimeout.current = setTimeout(() => {
        connect()
      }, 5000) // Reconnect after 5 seconds
    }
  }, [connectionStatus.connected, connectionStatus.reconnecting, connectionStatus.error, connect])

  return {
    // Data
    realtimeData,
    sessionData,
    
    // Connection status
    connectionStatus,
    
    // Actions
    connect,
    disconnect,
    sendCommand,
    
    // Computed values
    isConnected: connectionStatus.connected,
    isReconnecting: connectionStatus.reconnecting,
    hasError: !!connectionStatus.error
  }
}
