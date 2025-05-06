import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { WebSocketProvider } from "@/contexts/websocket-context"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "EEG Stress Monitor",
  description: "Real-time EEG-based brain state monitoring system",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
      <WebSocketProvider socketUrl="wss://muse-eeg-backend.onrender.com">
        {children}
      </WebSocketProvider>
      </body>
    </html>
  )
}
