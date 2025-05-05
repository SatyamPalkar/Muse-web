"use client"

import { useWebSocket } from "@/contexts/websocket-context"
import { ModeToggle } from "@/components/mode-toggle"
import { Activity, History, BarChart, User, Settings, Brain } from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { usePathname } from "next/navigation"

export function Sidebar() {
  const { connected } = useWebSocket()
  const pathname = usePathname()

  const routes = [
    {
      label: "Dashboard",
      icon: Activity,
      href: "/",
      active: pathname === "/",
    },
    {
      label: "Real-time EEG",
      icon: Brain,
      href: "/real-time",
      active: pathname === "/real-time",
    },
    {
      label: "History",
      icon: History,
      href: "/history",
      active: pathname === "/history",
    },
    {
      label: "Analytics",
      icon: BarChart,
      href: "/analytics",
      active: pathname === "/analytics",
    },
    {
      label: "Profile",
      icon: User,
      href: "/profile",
      active: pathname === "/profile",
    },
    {
      label: "Settings",
      icon: Settings,
      href: "/settings",
      active: pathname === "/settings",
    },
  ]

  return (
    <div className="flex h-screen flex-col border-r bg-background">
      <div className="flex h-14 items-center border-b px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Activity className="h-6 w-6 text-rose-500" />
          <span>StressWave</span>
        </Link>
      </div>
      <div className="flex-1 overflow-auto py-2">
        <nav className="grid items-start px-2 text-sm">
          {routes.map((route, index) => (
            <Link
              key={index}
              href={route.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-foreground",
                route.active && "bg-muted font-medium text-foreground",
              )}
            >
              <route.icon className="h-4 w-4" />
              {route.label}
            </Link>
          ))}
        </nav>
      </div>
      <div className="mt-auto border-t border-white/10 p-4 bg-gradient-to-r from-violet-900/30 to-indigo-900/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "h-3 w-3 rounded-full animate-pulse",
                connected
                  ? "bg-emerald-400 shadow-lg shadow-emerald-500/50"
                  : "bg-rose-400 shadow-lg shadow-rose-500/50",
              )}
            />
            <span className="text-xs font-medium">{connected ? "Connected" : "Disconnected"}</span>
          </div>
          <ModeToggle />
        </div>
      </div>
    </div>
  )
}
