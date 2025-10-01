/**
 * WebSocket client for real-time EEG data streaming
 */

interface EEGWebSocketMessage {
  type: 'realtime_analysis' | 'session_update' | 'status' | 'error'
  data: any
}

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

interface EEGSessionUpdate {
  timestamp: string
  samples_collected: number
  elapsed_time: number
  is_collecting: boolean
}

export class EEGWebSocketClient {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isConnecting = false
  private listeners: Map<string, Function[]> = new Map()

  constructor(private wsUrl: string = 'ws://localhost:8001/ws') {}

  /**
   * Connect to the WebSocket server
   */
  async connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return
    }

    this.isConnecting = true

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl)

        this.ws.onopen = () => {
          console.log('🧠 Connected to EEG WebSocket')
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.emit('connected')
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: EEGWebSocketMessage = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
            this.emit('error', { error: 'Failed to parse message' })
          }
        }

        this.ws.onclose = (event) => {
          console.log('🧠 EEG WebSocket disconnected:', event.code, event.reason)
          this.isConnecting = false
          this.emit('disconnected', { code: event.code, reason: event.reason })
          
          // Attempt to reconnect if not a manual close
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }

        this.ws.onerror = (error) => {
          console.error('🧠 EEG WebSocket error:', error)
          this.isConnecting = false
          this.emit('error', { error: 'WebSocket connection error' })
          reject(error)
        }
      } catch (error) {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect')
      this.ws = null
    }
  }

  /**
   * Send a command to the server
   */
  send(command: string, data?: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command, data }))
    } else {
      console.warn('WebSocket not connected, cannot send command:', command)
    }
  }

  /**
   * Add an event listener
   */
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(callback)
  }

  /**
   * Remove an event listener
   */
  off(event: string, callback: Function): void {
    const eventListeners = this.listeners.get(event)
    if (eventListeners) {
      const index = eventListeners.indexOf(callback)
      if (index > -1) {
        eventListeners.splice(index, 1)
      }
    }
  }

  /**
   * Emit an event to all listeners
   */
  private emit(event: string, data?: any): void {
    const eventListeners = this.listeners.get(event)
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error)
        }
      })
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: EEGWebSocketMessage): void {
    switch (message.type) {
      case 'realtime_analysis':
        this.emit('realtime', message.data as EEGRealtimeData)
        break
      case 'session_update':
        this.emit('session', message.data as EEGSessionUpdate)
        break
      case 'status':
        this.emit('status', message.data)
        break
      case 'error':
        this.emit('error', message.data)
        break
      default:
        console.warn('Unknown message type:', message.type)
    }
  }

  /**
   * Schedule a reconnection attempt
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) // Exponential backoff
    
    console.log(`🧠 Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`)
    
    setTimeout(() => {
      if (this.reconnectAttempts <= this.maxReconnectAttempts) {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error)
        })
      }
    }, delay)
  }

  /**
   * Get connection status
   */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  /**
   * Get connection state
   */
  get readyState(): number | null {
    return this.ws ? this.ws.readyState : null
  }
}

// React hook for WebSocket integration
export function useEEGWebSocket(wsUrl?: string) {
  const client = new EEGWebSocketClient(wsUrl)
  
  return {
    client,
    connect: () => client.connect(),
    disconnect: () => client.disconnect(),
    send: (command: string, data?: any) => client.send(command, data),
    on: (event: string, callback: Function) => client.on(event, callback),
    off: (event: string, callback: Function) => client.off(event, callback),
    isConnected: client.isConnected,
    readyState: client.readyState
  }
}
