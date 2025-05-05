import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Here you would typically process the data or store it
    // For this example, we'll just return it

    return NextResponse.json({
      success: true,
      message: "Data received successfully",
      data,
    })
  } catch (error) {
    return NextResponse.json({ success: false, message: "Failed to process data" }, { status: 500 })
  }
}
