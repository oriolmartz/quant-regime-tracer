function friendlySource(source: unknown) {
  const value = String(source || '')
  const normalized = value.toLowerCase()
  if (normalized.includes('uploaded_csv') || normalized.includes('csv')) return 'Uploaded CSV'
  if (normalized.includes('sample')) return 'Offline sample'
  if (normalized.includes('yfinance') || normalized.includes('cache:yfinance')) return 'Real market data · Yahoo Finance'
  return value || '—'
}

export default function DataQualityPanel({ quality }) {
  if (!quality) return null
  const tone = quality.status === 'strong'
    ? 'bg-[#EEF5F0] text-positive border-[#CFE0D5]'
    : quality.status === 'review'
      ? 'bg-[#FBF3E4] text-warning border-[#EAD8B6]'
      : 'bg-[#F7EEEE] text-danger border-[#E7CCCC]'

  return (
    <div className="card p-6">
      <div className="mb-5 flex items-start justify-between gap-4">
        <div>
          <p className="label">Data quality</p>
          <h3 className="mt-2 text-lg font-semibold text-ink">Input audit</h3>
          <p className="mt-2 text-sm leading-6 text-muted">Checks continuity, cache/source behavior and data sufficiency before interpreting regimes.</p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-xs font-semibold capitalize ${tone}`}>{quality.status}</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Source</span><strong className="mt-1 block text-ink">{friendlySource(quality.source)}</strong></div>
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Observations</span><strong className="mt-1 block text-ink">{quality.observations}</strong></div>
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Window</span><strong className="mt-1 block text-ink">{quality.date_start} → {quality.date_end}</strong></div>
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Largest gap</span><strong className="mt-1 block text-ink">{quality.largest_gap_days} days</strong></div>
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Annualization</span><strong className="mt-1 block text-ink">{quality.annualization_factor ?? '—'} periods/year</strong></div>
        <div className="rounded-2xl bg-ivory p-4"><span className="text-xs text-subdued">Observed calendar</span><strong className="mt-1 block text-ink">{String(quality.annualization_calendar || '—').replace(/_/g, ' ')}</strong></div>
      </div>

      {(quality.notes || []).length > 0 && (
        <ul className="mt-4 space-y-1 rounded-2xl border border-line bg-white p-4 text-xs leading-5 text-muted">
          {quality.notes.map((note, idx) => <li key={idx}>• {note}</li>)}
        </ul>
      )}
    </div>
  )
}
