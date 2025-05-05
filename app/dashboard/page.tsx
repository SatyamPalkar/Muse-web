import { Header } from "@/components/header"
import { CurrentStateDisplay } from "@/components/current-state-display"
import { BrainwaveActivity } from "@/components/brainwave-activity"
import { HistoricalTrend } from "@/components/historical-trend"
import { FeatureImportance } from "@/components/feature-importance"
import { SessionStatistics } from "@/components/session-statistics"

export default function Dashboard() {
  return (
    <div className="flex flex-col min-h-screen bg-black">
      <Header />
      <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-8">
          <CurrentStateDisplay />
          <HistoricalTrend />
          <SessionStatistics />
        </div>
        <div className="space-y-8">
          <BrainwaveActivity />
          <FeatureImportance />
        </div>
      </main>
    </div>
  )
}
