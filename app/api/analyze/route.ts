import { NextRequest, NextResponse } from "next/server"

// Configuration for Python backend
const EEG_API_BASE_URL = process.env.EEG_API_URL || 'http://localhost:8001'

interface EEGData {
  theta: number
  alpha: number
  beta: number
  gamma: number
  delta: number
  timestamp?: string
}

interface AnalysisResult {
  success: boolean
  data: {
    timestamp: string
    stress_level: string
    confidence: number
    overall_stress: number
    arousal_level: number
    temporal_trend: string
    recommendations: string[]
    stress_indicators: Record<string, number>
  }
}

async function fetchFromPythonBackend(endpoint: string, options: RequestInit = {}): Promise<any> {
  try {
    const response = await fetch(`${EEG_API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(5000),
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

function generateMockAnalysis(eegData: EEGData): AnalysisResult {
  // Simple mock analysis based on EEG values
  const beta_alpha_ratio = eegData.beta / eegData.alpha
  const stress_index = (eegData.beta + eegData.gamma * 1.5) / (eegData.alpha + eegData.theta * 0.5) - 1.0
  
  let stress_level = "Neutral"
  let confidence = 0.7
  
  if (stress_index > 0.6) {
    stress_level = "High Stress"
    confidence = 0.85
  } else if (stress_index > 0.3) {
    stress_level = "Moderate Stress"
    confidence = 0.8
  } else if (stress_index > 0.0) {
    stress_level = "Light Stress"
    confidence = 0.75
  } else if (stress_index < -0.2) {
    stress_level = "Relaxed"
    confidence = 0.8
  }

  const recommendations = []
  if (stress_level.includes("Stress")) {
    recommendations.push("Consider taking a short break")
    recommendations.push("Practice deep breathing exercises")
  } else if (stress_level === "Relaxed") {
    recommendations.push("Good mental state - continue current activity")
  }

  return {
    success: true,
    data: {
      timestamp: new Date().toISOString(),
      stress_level,
      confidence,
      overall_stress: Math.max(0, Math.min(1, stress_index + 0.5)),
      arousal_level: beta_alpha_ratio / 3,
      temporal_trend: "Stable",
      recommendations,
      stress_indicators: {
        beta_alpha_ratio,
        stress_index,
        theta_beta_ratio: eegData.theta / eegData.beta,
        alpha_beta_ratio: eegData.alpha / eegData.beta,
        gamma_beta_ratio: eegData.gamma / eegData.beta,
        total_power: eegData.theta + eegData.alpha + eegData.beta + eegData.gamma + eegData.delta
      }
    }
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Validate EEG data
    const requiredFields = ['theta', 'alpha', 'beta', 'gamma', 'delta']
    const missingFields = requiredFields.filter(field => !(field in body) || typeof body[field] !== 'number')
    
    if (missingFields.length > 0) {
      return NextResponse.json(
        { 
          success: false, 
          error: `Missing or invalid fields: ${missingFields.join(', ')}` 
        },
        { status: 400 }
      )
    }

    const eegData: EEGData = {
      theta: body.theta,
      alpha: body.alpha,
      beta: body.beta,
      gamma: body.gamma,
      delta: body.delta,
      timestamp: body.timestamp || new Date().toISOString()
    }

    // Check if we should use mock data (for development)
    const useMockData = process.env.NODE_ENV === 'development' && process.env.USE_MOCK_EEG === 'true'

    if (useMockData) {
      const mockResult = generateMockAnalysis(eegData)
      return NextResponse.json({
        ...mockResult,
        source: 'mock'
      })
    }

    // Try to analyze with Python backend
    try {
      const response = await fetchFromPythonBackend('/api/analyze', {
        body: JSON.stringify(eegData)
      })

      return NextResponse.json({
        ...response,
        source: 'backend'
      })
    } catch (backendError) {
      // Fallback to mock analysis if backend is unavailable
      console.warn('Backend unavailable, using mock analysis:', backendError)
      const mockResult = generateMockAnalysis(eegData)
      return NextResponse.json({
        ...mockResult,
        source: 'mock_fallback',
        error: 'Backend unavailable'
      })
    }
  } catch (error) {
    console.error('Error in analyze POST route:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: "Internal server error" 
      },
      { status: 500 }
    )
  }
}
