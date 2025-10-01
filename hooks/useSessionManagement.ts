"use client"

import { useState, useEffect, useCallback } from 'react'

interface SessionConfig {
  duration: number
  auto_start?: boolean
}

interface SessionResult {
  window_seconds: number
  sample_count: number
  dominant_state: string
  stress_level: string
  confidence: number
  beta_alpha_ratio: number
  stress_index: number
  temporal_trend: string
  evidence: string[]
  recommendations: string[]
  raw_metrics: Record<string, number>
}

interface SessionStatus {
  is_active: boolean
  duration: number
  samples_collected: number
  elapsed_time: number
}

export function useSessionManagement() {
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>({
    is_active: false,
    duration: 0,
    samples_collected: 0,
    elapsed_time: 0
  })
  
  const [sessionResults, setSessionResults] = useState<Record<string, SessionResult>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch session status from backend
  const fetchSessionStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/session?action=status')
      const data = await response.json()
      
      if (data.success) {
        setSessionStatus({
          is_active: data.data.is_collecting || false,
          duration: data.data.session_duration || 0,
          samples_collected: data.data.samples_collected || 0,
          elapsed_time: data.data.session_duration || 0
        })
      }
    } catch (error) {
      console.error('Failed to fetch session status:', error)
      setError('Failed to fetch session status')
    }
  }, [])

  // Start a new session
  const startSession = useCallback(async (config: SessionConfig = { duration: 120 }) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'start',
          config
        })
      })
      
      const data = await response.json()
      
      if (data.success) {
        setSessionStatus(prev => ({
          ...prev,
          is_active: true,
          duration: config.duration,
          samples_collected: 0,
          elapsed_time: 0
        }))
        
        // Start polling for status updates
        const interval = setInterval(() => {
          fetchSessionStatus()
        }, 2000) // Poll every 2 seconds
        
        // Store interval ID for cleanup
        ;(startSession as any).intervalId = interval
        
        return { success: true, message: data.message }
      } else {
        setError(data.message || 'Failed to start session')
        return { success: false, error: data.message }
      }
    } catch (error) {
      const errorMessage = 'Failed to start session'
      setError(errorMessage)
      console.error(errorMessage, error)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }, [fetchSessionStatus])

  // Stop current session
  const stopSession = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'stop'
        })
      })
      
      const data = await response.json()
      
      if (data.success) {
        // Clear any existing polling
        if ((startSession as any).intervalId) {
          clearInterval((startSession as any).intervalId)
          ;(startSession as any).intervalId = null
        }
        
        setSessionStatus(prev => ({
          ...prev,
          is_active: false
        }))
        
        // Update session results if available
        if (data.results) {
          setSessionResults(data.results)
        }
        
        return { success: true, message: data.message, results: data.results }
      } else {
        setError(data.message || 'Failed to stop session')
        return { success: false, error: data.message }
      }
    } catch (error) {
      const errorMessage = 'Failed to stop session'
      setError(errorMessage)
      console.error(errorMessage, error)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fetch session results
  const fetchSessionResults = useCallback(async () => {
    try {
      const response = await fetch('/api/session?action=results')
      const data = await response.json()
      
      if (data.success && data.data) {
        setSessionResults(data.data)
      }
    } catch (error) {
      console.error('Failed to fetch session results:', error)
      setError('Failed to fetch session results')
    }
  }, [])

  // Auto-stop session when duration is reached
  useEffect(() => {
    if (sessionStatus.is_active && sessionStatus.duration > 0 && sessionStatus.elapsed_time >= sessionStatus.duration) {
      stopSession()
    }
  }, [sessionStatus.is_active, sessionStatus.duration, sessionStatus.elapsed_time, stopSession])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if ((startSession as any).intervalId) {
        clearInterval((startSession as any).intervalId)
      }
    }
  }, [])

  return {
    // State
    sessionStatus,
    sessionResults,
    isLoading,
    error,
    
    // Actions
    startSession,
    stopSession,
    fetchSessionStatus,
    fetchSessionResults,
    
    // Computed values
    isActive: sessionStatus.is_active,
    progress: sessionStatus.duration > 0 ? (sessionStatus.elapsed_time / sessionStatus.duration) * 100 : 0,
    timeRemaining: Math.max(0, sessionStatus.duration - sessionStatus.elapsed_time)
  }
}
