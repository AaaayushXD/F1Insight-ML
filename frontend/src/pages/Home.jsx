import React from 'react'
import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div>
      <h1 className="text-3xl font-bold text-black mb-2">
        Race outcome prediction & <span className="text-accent">strategy assistance</span>
      </h1>
      <p className="text-h2 text-lg mb-8">
        F1Insight uses machine learning (Random Forest, XGBoost, Gradient Boosting) to predict finishing positions
        and podium probabilities, and Monte Carlo simulation to recommend pit-stop strategies.
      </p>
      <div className="grid md:grid-cols-3 gap-6">
        <Link
          to="/predictions"
          className="p-6 rounded-xl border-2 border-primary bg-surface hover:bg-primary hover:text-white transition-colors"
        >
          <h2 className="text-xl font-semibold text-h2 mb-2">Race predictions</h2>
          <p className="text-sm">Select a season and race to see predicted finish positions and podium probabilities for all drivers.</p>
        </Link>
        <Link
          to="/strategy"
          className="p-6 rounded-xl border-2 border-satellite bg-surface hover:bg-satellite hover:text-white transition-colors"
        >
          <h2 className="text-xl font-semibold text-h2 mb-2">Strategy</h2>
          <p className="text-sm">Enter expected position and uncertainty to get pit-stop strategy recommendations from Monte Carlo simulation.</p>
        </Link>
        <Link
          to="/compare"
          className="p-6 rounded-xl border-2 border-primary bg-surface hover:bg-primary hover:text-white transition-colors"
        >
          <h2 className="text-xl font-semibold text-h2 mb-2">Driver compare</h2>
          <p className="text-sm">Compare predictions for two drivers at a selected race.</p>
        </Link>
      </div>
      <p className="mt-8 text-sm text-gray-500">
        Data: Ergast API (2014–2025). No real-time telemetry. Models are explainable and trained with season-aware splits to avoid leakage.
      </p>
    </div>
  )
}
