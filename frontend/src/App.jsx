import React from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Home from './pages/Home'
import RacePredictions from './pages/RacePredictions'
import Strategy from './pages/Strategy'
import DriverCompare from './pages/DriverCompare'

function Nav() {
  const loc = useLocation()
  const link = (to, label) => (
    <Link
      to={to}
      className={`px-4 py-2 rounded-lg font-medium ${
        loc.pathname === to ? 'bg-primary text-white' : 'text-h2 hover:bg-surface'
      }`}
    >
      {label}
    </Link>
  )
  return (
    <nav className="border-b border-gray-200 bg-white">
      <div className="max-w-6xl mx-auto px-4 flex items-center justify-between h-14">
        <Link to="/" className="font-bold text-xl text-primary">F1Insight</Link>
        <div className="flex gap-2">
          {link('/', 'Home')}
          {link('/predictions', 'Race predictions')}
          {link('/strategy', 'Strategy')}
          {link('/compare', 'Driver compare')}
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predictions" element={<RacePredictions />} />
          <Route path="/strategy" element={<Strategy />} />
          <Route path="/compare" element={<DriverCompare />} />
        </Routes>
      </main>
    </div>
  )
}
