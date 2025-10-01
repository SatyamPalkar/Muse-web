import { NextRequest, NextResponse } from "next/server"

// Configuration for Python backend
const EEG_API_BASE_URL = process.env.EEG_API_URL || 'http://localhost:8001'

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

async function fetchFromPythonBackend(endpoint: string, options: RequestInit = {}): Promise<any> {
  try {
    const response = await fetch(`${EEG_API_BASE_URL}${endpoint}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000), // 10 second timeout for sessions
      ...options
    })

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error('Error fetching from Python backend:', error)
    throw error
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const action = searchParams.get('action') || 'results'

    if (action === 'results') {
      // Get session results
      const response = await fetchFromPythonBackend('/api/session/results')
      return NextResponse.json(response)
    } else if (action === 'status') {
      // Get session status
      const response = await fetchFromPythonBackend('/api/status')
      return NextResponse.json(response)
    } else {
      return NextResponse.json(
        { success: false, error: "Invalid action parameter" },
        { status: 400 }
      )
    }
  } catch (error) {
    console.error('Error in session GET route:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: "Backend unavailable",
        message: "Python backend is not running or accessible"
      },
      { status: 503 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, config } = body

    if (action === 'start') {
      // Start a new session
      const sessionConfig: SessionConfig = config || { duration: 120, auto_start: true }
      
      const response = await fetchFromPythonBackend('/api/session/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sessionConfig)
      })

      return NextResponse.json(response)
    } else if (action === 'stop') {
      // Stop current session
      const response = await fetchFromPythonBackend('/api/session/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      return NextResponse.json(response)
    } else {
      return NextResponse.json(
        { success: false, error: "Invalid action. Use 'start' or 'stop'" },
        { status: 400 }
      )
    }
  } catch (error) {
    console.error('Error in session POST route:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: "Backend unavailable",
        message: "Python backend is not running or accessible"
      },
      { status: 503 }
    )
  }
}
