import React, { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function DriverCompare() {
  const [seasons, setSeasons] = useState([])
  const [races, setRaces] = useState([])
  const [drivers, setDrivers] = useState([])
  const [season, setSeason] = useState(null)
  const [round, setRound] = useState(null)
  const [driverA, setDriverA] = useState('')
  const [driverB, setDriverB] = useState('')
  const [predA, setPredA] = useState(null)
  const [predB, setPredB] = useState(null)
  const [strategyA, setStrategyA] = useState(null)
  const [strategyB, setStrategyB] = useState(null)
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
    fetch(`${API}/api/races?season=${season}`)
      .then((r) => r.json())
      .then((d) => {
        setRaces(d.races || [])
        if (d.races?.length) setRound(d.races[0].round)
      })
    fetch(`${API}/api/drivers?season=${season}`)
      .then((r) => r.json())
      .then((d) => setDrivers(d.drivers || []))
      .catch(() => setDrivers([]))
  }, [season])

  useEffect(() => {
    setRound(null)
    if (season != null && races.length) setRound(races[0].round)
  }, [season, races])

  const runCompare = () => {
    if (!season || !round || !driverA || !driverB) {
      setError('Select season, race, and both drivers.')
      return
    }
    setLoading(true)
    setError(null)
    setPredA(null)
    setPredB(null)
    setStrategyA(null)
    setStrategyB(null)
    const base = `${API}/predict?season=${season}&round=${round}&driver_id=`
    Promise.all([
      fetch(base + encodeURIComponent(driverA)).then((r) => r.ok ? r.json() : null),
      fetch(base + encodeURIComponent(driverB)).then((r) => r.ok ? r.json() : null),
    ])
      .then(([a, b]) => {
        if (a?.error) throw new Error(a.message)
        if (b?.error) throw new Error(b.message)
        setPredA(a)
        setPredB(b)
        const promises = []
        if (a?.predicted_finish_position != null) {
          promises.push(
            fetch(`${API}/strategy?predicted_position_mean=${a.predicted_finish_position}&predicted_position_std=2`)
              .then((r) => r.json())
              .then(setStrategyA)
          )
        }
        if (b?.predicted_finish_position != null) {
          promises.push(
            fetch(`${API}/strategy?predicted_position_mean=${b.predicted_finish_position}&predicted_position_std=2`)
              .then((r) => r.json())
              .then(setStrategyB)
          )
        }
        return Promise.all(promises)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-h2 mb-6">Driver comparison</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="p-4 rounded-xl border border-gray-200">
          <label className="block text-sm font-medium text-h2 mb-2">Season / Race</label>
          <div className="flex gap-2 flex-wrap">
            <select
              value={season ?? ''}
              onChange={(e) => setSeason(Number(e.target.value))}
              className="border rounded-lg px-3 py-2"
            >
              {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select
              value={round ?? ''}
              onChange={(e) => setRound(Number(e.target.value))}
              className="border rounded-lg px-3 py-2 min-w-[180px]"
            >
              {races.map((r) => <option key={r.round} value={r.round}>{r.race_name}</option>)}
            </select>
          </div>
        </div>
        <div className="p-4 rounded-xl border border-gray-200">
          <label className="block text-sm font-medium text-h2 mb-2">Driver A / Driver B</label>
          <div className="flex gap-2 flex-wrap">
            <select
              value={driverA}
              onChange={(e) => setDriverA(e.target.value)}
              className="border rounded-lg px-3 py-2 flex-1 min-w-0"
            >
              <option value="">Select driver</option>
              {drivers.map((d) => (
                <option key={d.driver_id} value={d.driver_id}>{d.name ?? d.driver_id}</option>
              ))}
            </select>
            <select
              value={driverB}
              onChange={(e) => setDriverB(e.target.value)}
              className="border rounded-lg px-3 py-2 flex-1 min-w-0"
            >
              <option value="">Select driver</option>
              {drivers.map((d) => (
                <option key={d.driver_id} value={d.driver_id}>{d.name ?? d.driver_id}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
      <button
        onClick={runCompare}
        disabled={loading}
        className="bg-primary text-white font-medium py-2 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 mb-6"
      >
        {loading ? 'Loading…' : 'Compare'}
      </button>
      {error && (
        <div className="mb-4 p-4 rounded-lg bg-red-50 text-red-800 border border-red-200">{error}</div>
      )}
      {predA && predB && !loading && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-6 rounded-xl bg-surface border border-gray-200">
            <h2 className="font-semibold text-h2 mb-4">{driverA.replace(/_/g, ' ')}</h2>
            <p><strong>Predicted position:</strong> <span className="tabular-nums">{predA.predicted_finish_position}</span></p>
            <p><strong>Podium probability:</strong> <span className="tabular-nums">{predA.podium_probability != null ? `${(predA.podium_probability * 100).toFixed(1)}%` : '—'}</span></p>
            {strategyA?.best_strategy && (
              <p className="mt-2 text-sm text-gray-600">Best strategy expected pos: {strategyA.best_strategy.expected_position?.toFixed(2)}</p>
            )}
          </div>
          <div className="p-6 rounded-xl bg-surface border border-gray-200">
            <h2 className="font-semibold text-h2 mb-4">{driverB.replace(/_/g, ' ')}</h2>
            <p><strong>Predicted position:</strong> <span className="tabular-nums">{predB.predicted_finish_position}</span></p>
            <p><strong>Podium probability:</strong> <span className="tabular-nums">{predB.podium_probability != null ? `${(predB.podium_probability * 100).toFixed(1)}%` : '—'}</span></p>
            {strategyB?.best_strategy && (
              <p className="mt-2 text-sm text-gray-600">Best strategy expected pos: {strategyB.best_strategy.expected_position?.toFixed(2)}</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
