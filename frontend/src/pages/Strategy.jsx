import React, { useState } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Strategy() {
  const [mean, setMean] = useState(5)
  const [std, setStd] = useState(2)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchStrategy = () => {
    setLoading(true)
    setError(null)
    setResult(null)
    fetch(`${API}/strategy?predicted_position_mean=${mean}&predicted_position_std=${std}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText)
        return r.json()
      })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-h2 mb-2">Strategy recommendation</h1>
      <p className="text-gray-600 mb-6">
        Monte Carlo simulation ranks pit-stop strategies by expected finishing position (with degradation and traffic uncertainty).
      </p>
      <div className="max-w-md p-6 rounded-xl bg-surface border border-gray-200 mb-6">
        <label className="block mb-4">
          <span className="text-sm font-medium text-h2">Expected finishing position (mean)</span>
          <input
            type="number"
            min={1}
            max={20}
            step={0.5}
            value={mean}
            onChange={(e) => setMean(Number(e.target.value))}
            className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2"
          />
        </label>
        <label className="block mb-4">
          <span className="text-sm font-medium text-h2">Uncertainty (std)</span>
          <input
            type="number"
            min={0.1}
            max={10}
            step={0.1}
            value={std}
            onChange={(e) => setStd(Number(e.target.value))}
            className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2"
          />
        </label>
        <button
          onClick={fetchStrategy}
          disabled={loading}
          className="w-full bg-primary text-white font-medium py-2 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Get strategy'}
        </button>
      </div>
      {error && (
        <div className="mb-4 p-4 rounded-lg bg-red-50 text-red-800 border border-red-200">
          {error}
        </div>
      )}
      {result && !loading && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl border-2 border-primary bg-surface">
            <h2 className="font-semibold text-h2 mb-2">Best strategy</h2>
            <p className="text-sm text-gray-600">
              Expected position: <strong className="tabular-nums">{result.best_strategy?.expected_position?.toFixed(2)}</strong>
              {result.best_strategy?.typical_stops != null && (
                <> · Typical stops: <strong>{result.best_strategy.typical_stops}</strong></>
              )}
            </p>
          </div>
          {result.strategy_ranking?.length > 1 && (
            <div>
              <h2 className="font-semibold text-h2 mb-2">Full ranking</h2>
              <ul className="space-y-2">
                {result.strategy_ranking.map((s, i) => (
                  <li key={i} className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span>Rank {s.rank}: {s.label ?? `Strategy ${s.strategy_id}`}</span>
                    <span className="tabular-nums text-h2">Expected pos. {s.expected_position?.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
