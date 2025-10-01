import { NextRequest, NextResponse } from "next/server"

// Configuration for Python backend
const EEG_API_BASE_URL = process.env.EEG_API_URL || 'http://localhost:8001'
const USE_MOCK_DATA = process.env.NODE_ENV === 'development' && process.env.USE_MOCK_EEG === 'true'

interface EEGData {
  timestamp: string
  theta: number
  alpha: number
  beta: number
  gamma: number
  delta: number
  stress_level: string
  confidence: number
  recommendations: string[]
  overall_stress: number
  arousal_level: number
  temporal_trend: string
  stress_indicators: Record<string, number>
}

interface EEGAPIResponse {
  success: boolean
  data: EEGData
  connected?: boolean
}

async function fetchFromPythonBackend(endpoint: string): Promise<EEGAPIResponse> {
  try {
    const response = await fetch(`${EEG_API_BASE_URL}${endpoint}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add timeout for production
      signal: AbortSignal.timeout(5000)
    })

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Error fetching from Python backend:', error)
    throw error
  }
}

function generateMockData(): EEGData {
  return {
    timestamp: new Date().toISOString(),
    theta: 15 + Math.random() * 10,
    alpha: 25 + Math.random() * 15,
    beta: 20 + Math.random() * 20,
    gamma: 8 + Math.random() * 12,
    delta: 25 + Math.random() * 10,
    stress_level: "Moderate Stress",
    confidence: 0.85 + Math.random() * 0.1,
    recommendations: [
      "Consider taking a short break",
      "Practice deep breathing exercises",
      "Adjust your environment"
    ],
    overall_stress: 0.4 + Math.random() * 0.4,
    arousal_level: 0.3 + Math.random() * 0.3,
    temporal_trend: ["Increasing", "Decreasing", "Stable"][Math.floor(Math.random() * 3)],
    stress_indicators: {
      beta_alpha_ratio: 1.2 + Math.random() * 0.6,
      stress_index: 0.3 + Math.random() * 0.4,
      arousal_index: 0.2 + Math.random() * 0.3
    }
  }
}

export async function GET(request: NextRequest) {
  try {
    // Use mock data if configured or if backend is unavailable
    if (USE_MOCK_DATA) {
      const mockData = generateMockData()
      return NextResponse.json({
        success: true,
        data: mockData,
        connected: true,
        source: 'mock'
      })
    }

    // Try to fetch from Python backend
    try {
      const backendResponse = await fetchFromPythonBackend('/api/realtime')
      return NextResponse.json({
        ...backendResponse,
        connected: true,
        source: 'backend'
      })
    } catch (backendError) {
      // Fallback to mock data if backend is unavailable
      console.warn('Backend unavailable, using mock data:', backendError)
      const mockData = generateMockData()
      return NextResponse.json({
        success: true,
        data: mockData,
        connected: false,
        source: 'mock_fallback',
        error: 'Backend unavailable'
      })
    }
  } catch (error) {
    console.error('Error in EEG API route:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: "Internal server error",
        connected: false,
        source: 'error'
      },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  // Handle configuration updates or commands to the EEG analyzer
  try {
    const body = await request.json()
    
    // Here you would send commands to your Python backend
    // For example: start/stop analysis, change settings, etc.
    
    return NextResponse.json({
      success: true,
      message: "Command processed",
      data: body
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: "Invalid request" },
      { status: 400 }
    )
  }
}
