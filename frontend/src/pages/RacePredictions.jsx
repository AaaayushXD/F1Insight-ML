import React, { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function RacePredictions() {
  const [seasons, setSeasons] = useState([])
  const [races, setRaces] = useState([])
  const [season, setSeason] = useState(null)
  const [round, setRound] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/seasons`)
      .then((r) => r.json())
      .then((d) => {
        setSeasons(d.seasons || [])
        if (d.seasons?.length) setSeason(d.seasons[d.seasons.length - 1])
      })
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (season == null) return
    setRaces([])
    setRound(null)
    fetch(`${API}/api/races?season=${season}`)
      .then((r) => r.json())
      .then((d) => {
        setRaces(d.races || [])
        if (d.races?.length) setRound(d.races[0].round)
      })
      .catch((e) => setError(e.message))
  }, [season])

  useEffect(() => {
    if (season == null || round == null) return
    setLoading(true)
    setError(null)
    fetch(`${API}/api/predictions/race?season=${season}&round=${round}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText)
        return r.json()
      })
      .then((d) => setPredictions(d))
      .catch((e) => {
        setError(e.message)
        setPredictions(null)
      })
      .finally(() => setLoading(false))
  }, [season, round])

  return (
    <div>
      <h1 className="text-2xl font-bold text-h2 mb-6">Race predictions</h1>
      <div className="flex flex-wrap gap-4 items-end mb-6">
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium text-h2">Season</span>
          <select
            value={season ?? ''}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="border border-gray-300 rounded-lg px-3 py-2 min-w-[120px]"
          >
            {seasons.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium text-h2">Race</span>
          <select
            value={round ?? ''}
            onChange={(e) => setRound(Number(e.target.value))}
            className="border border-gray-300 rounded-lg px-3 py-2 min-w-[220px]"
          >
            {races.map((r) => (
              <option key={r.round} value={r.round}>{r.race_name} (R{r.round})</option>
            ))}
          </select>
        </label>
      </div>
      {error && (
        <div className="mb-4 p-4 rounded-lg bg-red-50 text-red-800 border border-red-200">
          {error}. Ensure the backend is running (uvicorn app.main:app --reload) and models are trained.
        </div>
      )}
      {loading && <p className="text-gray-500">Loading predictions…</p>}
      {predictions && !loading && (
        <>
          {predictions.predictions?.length === 0 && (
            <div className="mb-4 p-4 rounded-lg bg-surface border border-gray-200 text-h2">
              {predictions.message || "No prediction data for this race. Result/qualifying data may not have been collected for this (season, round)."}
            </div>
          )}
          <div className="overflow-x-auto">
          <table className="w-full border border-gray-200 rounded-xl overflow-hidden">
            <thead>
              <tr className="bg-primary text-white">
                <th className="text-left px-4 py-3">#</th>
                <th className="text-left px-4 py-3">Driver</th>
                <th className="text-left px-4 py-3">Constructor</th>
                <th className="text-right px-4 py-3">Predicted position</th>
                <th className="text-right px-4 py-3">Podium probability</th>
              </tr>
            </thead>
            <tbody>
              {predictions.predictions.map((p, i) => (
                <tr key={p.driver_id} className={i % 2 ? 'bg-surface/50' : ''}>
                  <td className="px-4 py-3 font-medium text-satellite">{i + 1}</td>
                  <td className="px-4 py-3">{p.driver_id.replace(/_/g, ' ')}</td>
                  <td className="px-4 py-3 text-gray-600">{p.constructor_id?.replace(/_/g, ' ') ?? '—'}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{p.predicted_finish_position ?? '—'}</td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {p.podium_probability != null ? `${(p.podium_probability * 100).toFixed(1)}%` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        </>
      )}
    </div>
  )
}
