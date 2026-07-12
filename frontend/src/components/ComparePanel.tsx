import { translateRegimeLabel, translateRiskBand } from '../lib/i18n'
import type { CompareResult, LanguageCode } from '../lib/types'
import type { Dispatch, SetStateAction } from 'react'

function pct(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

function toneForRisk(score = 0): string {
  if (score >= 0.66) return 'bg-[#F7EEEE] text-danger border-[#E7CCCC]'
  if (score >= 0.38) return 'bg-[#FBF3E4] text-warning border-[#EAD8B6]'
  return 'bg-[#EEF5F0] text-positive border-[#CFE0D5]'
}

function friendlySource(source: unknown, language: LanguageCode) {
  const value = String(source || '')
  const normalized = value.toLowerCase()
  if (normalized.includes('uploaded_csv') || normalized.includes('csv')) {
    return language === 'es' ? 'CSV subido' : 'Uploaded CSV'
  }
  if (normalized.includes('sample')) {
    return language === 'es' ? 'Muestra offline' : 'Offline sample'
  }
  if (normalized.includes('yfinance') || normalized.includes('cache:yfinance')) {
    return language === 'es' ? 'Datos reales · Yahoo Finance' : 'Real market data · Yahoo Finance'
  }
  return value || '—'
}

const copy = {
  en: {
    kicker: 'Cross-asset review',
    title: 'Portfolio-level regime comparison',
    body: 'The comparison layer runs the same HMM + Markov regime engine across several assets and ranks them by current review priority. This is a triage view, not a trading recommendation.',
    comparing: 'Comparing…',
    compare: 'Compare',
    assets: 'assets',
    highest: 'Highest review priority',
    average: 'Average risk score',
    lowest: 'Lowest risk score',
    ranked: 'Ranked summaries',
    currentByAsset: 'Current regime by asset',
    asset: 'Asset',
    regime: 'Regime',
    risk: 'Risk',
    stressTransition: 'Stress transition',
    drawdown: 'Drawdown',
    baseline: 'Baseline',
    data: 'Data',
    confidence: 'evidence',
    memo: 'Comparison memo',
  },
  es: {
    kicker: 'Revisión multi-activo',
    title: 'Comparación de regímenes a nivel portfolio',
    body: 'La capa de comparación ejecuta el mismo motor HMM + Markov sobre varios activos y los ordena por prioridad actual de revisión. Es una vista de triage, no una recomendación de trading.',
    comparing: 'Comparando…',
    compare: 'Comparar',
    assets: 'activos',
    highest: 'Mayor prioridad de revisión',
    average: 'Score de riesgo medio',
    lowest: 'Menor score de riesgo',
    ranked: 'Resumen ordenado',
    currentByAsset: 'Régimen actual por activo',
    asset: 'Activo',
    regime: 'Régimen',
    risk: 'Riesgo',
    stressTransition: 'Transición a estrés',
    drawdown: 'Drawdown',
    baseline: 'Baseline',
    data: 'Datos',
    confidence: 'evidencia',
    memo: 'Memo comparativo',
  },
}

type CopyKey = keyof typeof copy.en

function L(language: LanguageCode, key: CopyKey): string {
  return (copy[language] || copy.en)[key] || copy.en[key] || key
}

interface ComparePanelProps {
  assets: string[]
  selectedAssets: string[]
  setSelectedAssets: Dispatch<SetStateAction<string[]>>
  result?: CompareResult | null
  onRun: () => void
  loading: boolean
  language?: LanguageCode
}

export default function ComparePanel({ assets, selectedAssets, setSelectedAssets, result, onRun, loading, language = 'en' }: ComparePanelProps) {
  const lang = language
  function toggleAsset(asset: string) {
    setSelectedAssets((current: string[]) => {
      if (current.includes(asset)) return current.filter((item: string) => item !== asset)
      if (current.length >= 8) return current
      return [...current, asset]
    })
  }

  return (
    <section className="space-y-6">
      <div className="card p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="label">{L(lang, 'kicker')}</p>
            <h3 className="mt-2 text-xl font-semibold text-ink">{L(lang, 'title')}</h3>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">{L(lang, 'body')}</p>
          </div>
          <button onClick={onRun} disabled={loading || selectedAssets.length === 0} className="inline-flex h-11 items-center justify-center rounded-xl bg-brand px-5 text-sm font-semibold text-white transition hover:bg-[#0E172B] disabled:cursor-not-allowed disabled:opacity-60">
            {loading ? L(lang, 'comparing') : `${L(lang, 'compare')} ${selectedAssets.length} ${L(lang, 'assets')}`}
          </button>
        </div>

        <div className="mt-6 flex flex-wrap gap-2">
          {(assets || []).map((asset) => {
            const active = selectedAssets.includes(asset)
            return (
              <button key={asset} onClick={() => toggleAsset(asset)} className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${active ? 'border-[#BFD8D3] bg-soft text-accent' : 'border-line bg-white text-muted hover:border-[#BFD8D3] hover:text-accent'}`}>
                {asset}
              </button>
            )
          })}
        </div>
      </div>

      {result && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="card p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{L(lang, 'highest')}</p>
              <div className="mt-2 text-2xl font-semibold text-ink">{result.highest_risk_asset || '—'}</div>
            </div>
            <div className="card p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{L(lang, 'average')}</p>
              <div className="mt-2 text-2xl font-semibold text-ink">{pct(result.average_risk_score)}</div>
            </div>
            <div className="card p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{L(lang, 'lowest')}</p>
              <div className="mt-2 text-2xl font-semibold text-ink">{result.lowest_risk_asset || '—'}</div>
            </div>
          </div>

          <div className="card overflow-hidden">
            <div className="border-b border-line p-5">
              <p className="label">{L(lang, 'ranked')}</p>
              <h3 className="mt-2 text-lg font-semibold text-ink">{L(lang, 'currentByAsset')}</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line text-sm">
                <thead className="bg-ivory text-left text-xs uppercase tracking-[0.14em] text-subdued">
                  <tr>
                    <th className="px-5 py-3">{L(lang, 'asset')}</th>
                    <th className="px-5 py-3">{L(lang, 'regime')}</th>
                    <th className="px-5 py-3">{L(lang, 'risk')}</th>
                    <th className="px-5 py-3">{L(lang, 'stressTransition')}</th>
                    <th className="px-5 py-3">{L(lang, 'drawdown')}</th>
                    <th className="px-5 py-3">{L(lang, 'baseline')}</th>
                    <th className="px-5 py-3">{L(lang, 'data')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line bg-white">
                  {(result.summaries || []).map((row) => (
                    <tr key={row.asset}>
                      <td className="px-5 py-4 font-semibold text-ink">{row.asset}</td>
                      <td className="px-5 py-4 text-muted">{translateRegimeLabel(row.current_regime, lang)}<div className="text-xs text-subdued">{pct(row.evidence_strength ?? row.confidence)} {L(lang, 'confidence')}</div></td>
                      <td className="px-5 py-4"><span className={`rounded-full border px-2.5 py-1 text-xs font-semibold capitalize ${toneForRisk(row.risk_score)}`}>{pct(row.risk_score)} · {translateRiskBand(row.risk_band, lang)}</span></td>
                      <td className="px-5 py-4 text-muted">{pct(row.stress_transition_probability)}</td>
                      <td className="px-5 py-4 text-muted">{pct(row.latest_drawdown)}</td>
                      <td className="px-5 py-4 text-muted">{pct(row.baseline_agreement)} <span className="text-xs text-subdued">{row.baseline_verdict}</span></td>
                      <td className="px-5 py-4 text-muted">{row.data_quality_status}<div className="text-xs text-subdued">{friendlySource(row.source, lang)}</div><div className="text-xs text-subdued">{row.annualization_factor ?? '—'}/yr · {String(row.annualization_calendar || 'inferred').replace(/_/g, ' ')}</div></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-2xl border border-[#BFD8D3] bg-soft p-5 text-sm leading-6 text-brand">
            <div className="font-semibold">{L(lang, 'memo')}</div>
            <p className="mt-2">{result.portfolio_memo?.summary || '—'}</p>
            <ul className="mt-3 list-disc space-y-1 pl-5">
              {(result.portfolio_memo?.review_notes || []).map((note: string, idx: number) => <li key={idx}>{note}</li>)}
            </ul>
          </div>
        </>
      )}
    </section>
  )
}
