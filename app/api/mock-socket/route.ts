import { type NextRequest, NextResponse } from "next/server"
import { generateMockEEGData } from "@/utils/mock-data"

// This is a simple endpoint to test the WebSocket connection
// In a real application, you would use a proper WebSocket server
export async function GET(request: NextRequest) {
  return NextResponse.json({
    message:
      "This endpoint is for testing WebSocket connections. In a real application, you would connect to a WebSocket server.",
    sampleData: generateMockEEGData(),
  })
}
