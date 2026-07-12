import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent } from 'react'
import { Activity, AlertTriangle, ArrowRight, BarChart3, BookOpen, CheckCircle2, Copy, Database, Download, FileText, FileUp, GitBranch, Layers, RefreshCw, ShieldCheck } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { analyzeAsset, compareAssetSet, fetchAssets, fetchCaseStudy, fetchHealth, fetchProjectCard, fetchTimeWindows, uploadCsv } from './lib/api'
import { LANGUAGES, getTranslator, translateRegimeLabel, translateRiskBand } from './lib/i18n'
import MetricCard from './components/MetricCard'
import MemoPanel from './components/MemoPanel'
import PriceRegimeChart from './components/PriceRegimeChart'
import TransitionMatrix from './components/TransitionMatrix'
import VolatilityChart from './components/VolatilityChart'
import DrawdownChart from './components/DrawdownChart'
import RegimeProbabilityStrip from './components/RegimeProbabilityStrip'
import DiagnosticsPanel from './components/DiagnosticsPanel'
import BenchmarkPanel from './components/BenchmarkPanel'
import TimelineSegments from './components/TimelineSegments'
import ModelCardPanel from './components/ModelCardPanel'
import DataQualityPanel from './components/DataQualityPanel'
import BaselineComparisonPanel from './components/BaselineComparisonPanel'
import CaseStudyPanel from './components/CaseStudyPanel'
import ExportPanel from './components/ExportPanel'
import ComparePanel from './components/ComparePanel'
import PipelinePanel from './components/PipelinePanel'
import FeatureBehaviorChart from './components/FeatureBehaviorChart'
import RegimeStatsChart from './components/RegimeStatsChart'
import RegimeConfidenceChart from './components/RegimeConfidenceChart'
import RegimeSummaryMetrics from './components/RegimeSummaryMetrics'
import ModelEvaluationPanel from './components/ModelEvaluationPanel'
import RegimeTracebackPanel from './components/RegimeTracebackPanel'
import type { AnalysisResult, CaseStudy, CompareResult, DataMode, LanguageCode, ProjectCard, RiskTone, RegimeStats } from './lib/types'

function pct(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

function riskTone(score = 0): RiskTone {
  if (score >= 0.66) return 'red'
  if (score >= 0.38) return 'amber'
  return 'green'
}

function riskBandLabel(score = 0, language: LanguageCode = 'en') {
  if (score >= 0.66) return language === 'es' ? 'Elevado' : 'Elevated'
  if (score >= 0.38) return language === 'es' ? 'Moderado' : 'Moderate'
  return language === 'es' ? 'Contenido' : 'Contained'
}

function downloadBlob(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.style.display = 'none'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 250)
}

function reportFilename(result: AnalysisResult) {
  return `quantregimetracer_${result.asset}_${result.end || 'report'}.md`
}

function jsonFilename(result: AnalysisResult) {
  return `quantregimetracer_${result.asset}_${result.end || 'analysis'}.json`
}

function sourceLabel(result: AnalysisResult | null | undefined, t: (key: string) => string) {
  const source = String(result?.source_report?.source || result?.source || '').toLowerCase()
  if (!source) return '—'
  if (source.includes('uploaded_csv') || source.includes('csv')) return t('sourceCsv')
  if (source.includes('sample')) return t('sourceSample')
  if (source.includes('yfinance') || source.includes('cache:yfinance')) return t('sourceLive')
  return source
}

export default function App() {
  const [language, setLanguage] = useState<LanguageCode>(() => (localStorage.getItem('quantregimetracer_language') as LanguageCode) || 'en')
  const t = useMemo(() => getTranslator(language), [language])
  const [assets, setAssets] = useState(['SPY', 'QQQ', 'BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'NVDA', 'META', 'IWM', 'DIA', 'GLD', 'TLT'])
  const [windows, setWindows] = useState(['6M', '1Y', '2Y', '3Y', '5Y', 'MAX'])
  const [asset, setAsset] = useState('SPY')
  const [dataMode, setDataMode] = useState<DataMode>('real')
  const [forceRefresh, setForceRefresh] = useState(false)
  const [interval, setInterval] = useState('5Y')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [nRegimes, setNRegimes] = useState(3)
  const [apiHealth, setApiHealth] = useState<{ status?: string; version?: string } | null>(null)
  const [caseStudy, setCaseStudy] = useState<CaseStudy | null>(null)
  const [projectCard, setProjectCard] = useState<ProjectCard | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [comparison, setComparison] = useState<CompareResult | null>(null)
  const [selectedCompareAssets, setSelectedCompareAssets] = useState(['SPY', 'QQQ', 'BTC-USD'])
  const [activeTab, setActiveTab] = useState('dashboard')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [notice, setNotice] = useState<{ id: number; message: string } | null>(null)

  function showNotice(message: string) {
    const id = Date.now()
    setNotice({ id, message })
    window.setTimeout(() => {
      setNotice((current) => current?.id === id ? null : current)
    }, 2200)
  }

  function handleDownloadReport() {
    if (!result?.report_markdown) return
    downloadBlob(reportFilename(result), result.report_markdown, 'text/markdown;charset=utf-8')
    showNotice(t('reportDownloaded'))
  }

  function handleDownloadJson() {
    if (!result) return
    downloadBlob(jsonFilename(result), JSON.stringify(result, null, 2), 'application/json;charset=utf-8')
    showNotice(t('jsonDownloaded'))
  }

  async function runAnalysis(selectedAsset = asset) {
    setLoading(true)
    setError('')
    try {
      const data = await analyzeAsset({
        asset: selectedAsset,
        start: start || null,
        end: end || null,
        interval,
        n_regimes: Number(nRegimes),
        data_mode: dataMode,
        force_refresh: forceRefresh,
        language,
      })
      setResult(data)
      setActiveTab('dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errorGeneric'))
    } finally {
      setLoading(false)
    }
  }

  async function handleCsvUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const data = await uploadCsv(file, Number(nRegimes), language)
      setResult(data)
      setActiveTab('dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : t('csvFailed'))
    } finally {
      setLoading(false)
      event.target.value = ''
    }
  }


  async function runComparison() {
    setLoading(true)
    setError('')
    try {
      const data = await compareAssetSet({
        assets: selectedCompareAssets,
        start: start || null,
        end: end || null,
        interval,
        n_regimes: Number(nRegimes),
        data_mode: dataMode,
        force_refresh: forceRefresh,
        language,
      })
      setComparison(data)
      setActiveTab('compare')
    } catch (err) {
      setError(err instanceof Error ? err.message : t('comparisonFailed'))
    } finally {
      setLoading(false)
    }
  }

  function copyPortfolioSummary() {
    if (!projectCard && !result) return
    const summary = `${projectCard?.name || 'QuantRegimeTracer'} — ${projectCard?.tagline || 'Time-series regime intelligence'}\n\n${projectCard?.description || ''}\n\n${t('portfolioSummaryLatest')}: ${result?.asset || asset}, ${translateRegimeLabel(result?.current_regime?.label || 'regime analysis', language)}, ${pct(result?.current_regime?.evidence_strength as number | undefined)} ${t('evidenceStrength').toLowerCase()}.`
    navigator.clipboard?.writeText(summary)
    setCopied(true)
    showNotice(t('summaryCopied'))
    window.setTimeout(() => setCopied(false), 1600)
  }

  useEffect(() => {
    fetchAssets().then((d) => setAssets(d.assets || assets)).catch(() => {})
    fetchTimeWindows().then((d) => setWindows(d.windows || windows)).catch(() => {})
    fetchHealth().then(setApiHealth).catch(() => {})
    fetchCaseStudy().then(setCaseStudy).catch(() => {})
    fetchProjectCard().then(setProjectCard).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    localStorage.setItem('quantregimetracer_language', language)
  }, [language])

  const currentTone = useMemo(() => {
    const label = result?.current_regime?.label?.toLowerCase() || ''
    if (label.includes('stress')) return 'red'
    if (label.includes('expansion')) return 'green'
    return 'amber'
  }, [result])

  const riskBand = useMemo(() => riskBandLabel(result?.current_regime?.risk_score || 0, language), [result, language])

  const tabs: Array<[string, string, LucideIcon]> = [
    ['dashboard', t('dashboard'), Activity],
    ['traceback', t('traceback'), GitBranch],
    ['validation', t('validation'), Layers],
    ['compare', t('compare'), BarChart3],
    ['case', t('caseStudy'), BookOpen],
    ['export', t('export'), FileText],
  ]

  return (
    <div className="min-h-screen">
      <header className="relative overflow-hidden border-b border-white/10 bg-dark text-white">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(42,111,104,.22),transparent_38%),linear-gradient(135deg,#14213D_0%,#111827_54%,#0B1120_100%)]" />
        <nav className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent shadow-lg shadow-black/10">
              <Activity size={20} />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="text-base font-semibold tracking-tight text-white">QuantRegimeTracer</span>
                <span className="hidden text-stone-300/60 sm:inline">·</span>
                <span className="text-sm font-medium text-[#BFD8D3]">Built by Oriol Martínez</span>
              </div>
              <div className="text-[11px] text-stone-200/70">{t('subtitleMeta')}</div>
            </div>
          </div>
          <div className="hidden items-center gap-3 sm:flex">
            <span className="rounded-full border border-emerald-300/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
              API {apiHealth?.status || 'checking'} · {apiHealth?.version || '0.10.0'}
            </span>
            <div className="flex rounded-full border border-white/10 bg-white/[0.04] p-1">
              {LANGUAGES.map((item) => (
                <button key={item.code} onClick={() => setLanguage(item.code)} className={`rounded-full px-3 py-1 text-xs font-semibold transition ${language === item.code ? 'bg-white text-slate-950' : 'text-stone-100 hover:text-white'}`}>
                  {item.label}
                </button>
              ))}
            </div>
            <button onClick={handleDownloadReport} className="rounded-full border border-white/10 px-4 py-2 text-xs font-semibold text-stone-100 transition hover:border-[#BFD8D3]/60">{t('export')} memo</button>
            <button onClick={handleDownloadJson} className="rounded-full border border-white/10 px-4 py-2 text-xs font-semibold text-stone-100 transition hover:border-[#BFD8D3]/60">{t('export')} JSON</button>
          </div>
        </nav>

        <section className="relative z-10 mx-auto grid max-w-7xl gap-10 px-5 py-10 lg:grid-cols-[1.05fr_.95fr] lg:px-8 lg:py-12">
          <div>
            <div className="mb-5 inline-flex rounded-full border border-[#BFD8D3]/25 bg-[#2A6F68]/20 px-3 py-1 text-xs font-semibold text-[#D7E8E4]">
              {t('edition')}
            </div>
            <h1 className="max-w-4xl text-4xl font-semibold tracking-[-0.04em] text-white sm:text-5xl lg:text-6xl">
              {t('heroTitle')}
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-8 text-slate-300">
              {t('heroBody')}
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <button onClick={() => runAnalysis()} className="inline-flex items-center gap-2 rounded-xl bg-brand px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-black/10 transition hover:bg-[#0E172B]">
                {t('runAnalysis')} <ArrowRight size={16} />
              </button>
              <span className="inline-flex items-center gap-2 text-xs text-stone-300"><ShieldCheck size={15} /> {t('notAdvice')}</span>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 shadow-2xl shadow-black/10 backdrop-blur">
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-200/70">{t('currentRegime')}</p>
                <h2 className="mt-2 text-2xl font-semibold text-white">{result ? translateRegimeLabel(result.current_regime?.label, language) : t('awaitingAnalysis')}</h2>
              </div>
              <span className="rounded-full border border-[#BFD8D3]/25 bg-[#2A6F68]/20 px-3 py-1 text-xs font-semibold text-[#D7E8E4]">{result ? `${riskBand} ${t('risk')}` : t('ready')}</span>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <HeroStat label={t('evidenceStrength')} value={pct(result?.current_regime?.evidence_strength as number | undefined)} />
              <HeroStat label={t('riskScore')} value={pct(result?.current_regime?.risk_score)} />
              <HeroStat label={t('baselineAgreement')} value={pct(result?.baseline?.stress_agreement)} />
              <HeroStat label={t('source')} value={sourceLabel(result, t)} />
            </div>
          </div>
        </section>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-8 lg:px-8">
        <section className="card mb-6 p-4">
          <div className="grid gap-x-4 gap-y-2 md:grid-cols-[1fr_.7fr_.8fr_.7fr_auto] md:items-end">
            <div className="md:col-start-1 md:row-start-1">
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{t('asset')}</label>
              <select className="input h-11 w-full" value={asset} onChange={(e) => setAsset(e.target.value)}>
                {assets.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <div className="md:col-start-2 md:row-start-1">
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{t('window')}</label>
              <select className="input h-11 w-full" value={interval} onChange={(e) => setInterval(e.target.value)}>
                {windows.map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </div>
            <div className="flex flex-col md:col-start-3 md:row-start-1">
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{t('customStartDate')}</label>
              <input className="input h-11 w-full" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
            </div>
            <p className="pl-3 text-left text-[11px] leading-4 text-subdued md:col-start-3 md:row-start-2 md:justify-self-start">
              {t('customStartHelp')}
            </p>
            <div className="md:col-start-4 md:row-start-1">
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-subdued">{t('regimes')}</label>
              <select className="input h-11 w-full min-w-[105px]" value={nRegimes} onChange={(e) => setNRegimes(Number(e.target.value))}>
                {[2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <button disabled={loading} onClick={() => runAnalysis()} className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-brand px-5 text-sm font-semibold text-white transition hover:bg-[#0E172B] disabled:cursor-not-allowed disabled:opacity-60 md:col-start-5 md:row-start-1">
              {loading ? <RefreshCw className="animate-spin" size={16} /> : <Database size={16} />}
              {loading ? t('analyzing') : t('analyze')}
            </button>
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-[1fr_auto_auto_auto] md:items-center">
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex h-11 items-center gap-3 rounded-xl border border-line bg-ivory px-4 text-sm font-medium text-muted">
                <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subdued">{t('dataMode')}</span>
                <select className="bg-transparent text-sm font-semibold text-ink outline-none" value={dataMode} onChange={(e) => setDataMode(e.target.value as DataMode)}>
                  <option value="real">{t('dataModeReal')}</option>
                  <option value="auto">{t('dataModeAuto')}</option>
                  <option value="sample">{t('dataModeSample')}</option>
                </select>
              </label>
              <label className="flex h-11 items-center gap-3 rounded-xl border border-line bg-ivory px-4 text-sm font-medium text-muted">
                <input type="checkbox" checked={forceRefresh} disabled={dataMode === 'sample'} onChange={(e) => setForceRefresh(e.target.checked)} />
                {t('forceRefresh')}
              </label>
              <label className="inline-flex h-11 cursor-pointer items-center gap-2 rounded-xl border border-line bg-white px-4 text-sm font-semibold text-muted transition hover:border-[#BFD8D3] hover:text-brand">
                <FileUp size={16} /> {t('uploadCsv')}
                <input type="file" accept=".csv" className="hidden" onChange={handleCsvUpload} />
              </label>
              <span className="text-xs text-subdued">{t('csvHelp')}</span>
            </div>
            <button onClick={handleDownloadReport} disabled={!result} className="inline-flex h-11 items-center gap-2 rounded-xl border border-line bg-white px-4 text-sm font-semibold text-muted transition hover:border-[#BFD8D3] hover:text-brand disabled:opacity-50">
              <FileText size={16} /> {t('markdown')}
            </button>
            <button onClick={handleDownloadJson} disabled={!result} className="inline-flex h-11 items-center gap-2 rounded-xl border border-line bg-white px-4 text-sm font-semibold text-muted transition hover:border-[#BFD8D3] hover:text-brand disabled:opacity-50">
              <Download size={16} /> {t('json')}
            </button>
            <button onClick={copyPortfolioSummary} disabled={!result} className="inline-flex h-11 items-center gap-2 rounded-xl border border-line bg-white px-4 text-sm font-semibold text-muted transition hover:border-[#BFD8D3] hover:text-brand disabled:opacity-50">
              <Copy size={16} /> {copied ? t('copied') : t('summary')}
            </button>
          </div>
          {error && <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div>}
          {result?.warnings?.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-800">
              <div className="mb-1 flex items-center gap-2 font-semibold"><AlertTriangle size={14} /> {t('warnings')}</div>
              {result.warnings.slice(0, 5).map((w, i) => <div key={i}>• {w}</div>)}
            </div>
          )}
          {result?.source_report && (
            <div className="mt-4 grid gap-3 rounded-xl border border-line bg-white px-4 py-3 text-xs text-muted md:grid-cols-4">
              <div><span className="font-semibold text-ink">{t('dataMode')}:</span> {String(result.source_report.mode || '—')}</div>
              <div><span className="font-semibold text-ink">{t('source')}:</span> {sourceLabel(result, t)}</div>
              <div><span className="font-semibold text-ink">{t('realBacked')}:</span> {result.source_report.is_real_data ? t('yes') : t('no')}</div>
              <div><span className="font-semibold text-ink">{t('actualWindow')}:</span> {result.source_report.actual_start || '—'} → {result.source_report.actual_end || '—'}</div>
            </div>
          )}
        </section>

        <div className="mb-6 flex flex-wrap gap-2 rounded-2xl border border-line bg-white p-2 shadow-premium">
          {tabs.map(([id, label, Icon]) => (
            <button key={id} onClick={() => setActiveTab(id)} className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition ${activeTab === id ? 'bg-brand text-white shadow-sm shadow-black/5' : 'text-muted hover:bg-ivory hover:text-ink'}`}>
              <Icon size={16} /> {label}
            </button>
          ))}
        </div>

        {!result && !['compare', 'case'].includes(activeTab) && (
          <section className="card mb-10 flex min-h-[220px] flex-col items-center justify-center px-6 py-10 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-soft text-brand">
              <Database size={22} />
            </span>
            <h2 className="mt-4 text-xl font-semibold text-ink">{t('analysisPromptTitle')}</h2>
            <p className="mt-2 max-w-xl text-sm leading-6 text-muted">{t('analysisPromptBody')}</p>
          </section>
        )}

        {result && activeTab === 'dashboard' && (
          <>
            <section className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-6">
              <MetricCard label={t('currentRegime')} value={translateRegimeLabel(result.current_regime.label, language)} detail={String(result.current_regime.assignment_type || t('stateAssignment'))} tone={currentTone} />
              <MetricCard label={t('evidenceStrength')} value={pct(result.current_regime.evidence_strength as number | undefined)} detail={t('traceabilityScore')} tone={riskTone(1 - Number(result.current_regime.evidence_strength || 0))} />
              <MetricCard label={t('riskScore')} value={pct(result.current_regime.risk_score)} detail={riskBand} tone={riskTone(result.current_regime.risk_score)} />
              <MetricCard label={t('stayProbability')} value={pct(result.current_regime.stay_probability)} detail={t('markovPersistence')} tone="blue" />
              <MetricCard label={t('stressTransition')} value={pct(result.current_regime.stress_transition_probability)} detail={t('nextStateRisk')} tone={result.current_regime.stress_transition_probability > 0.25 ? 'red' : 'amber'} />
              <MetricCard label={t('baselineAgreement')} value={pct(result.baseline?.stress_agreement)} detail={result.baseline?.verdict || t('validation')} tone="neutral" />
            </section>

            <section className="mb-6 rounded-2xl border border-[#BFD8D3] bg-soft p-5 text-sm leading-6 text-brand">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="font-semibold text-ink">{t('whyThisRegime')}</div>
                  <p className="mt-1 max-w-3xl text-muted">{t('whyThisRegimeBody')}</p>
                </div>
                <button onClick={() => setActiveTab('traceback')} className="inline-flex h-10 items-center justify-center gap-2 rounded-xl bg-brand px-4 text-sm font-semibold text-white transition hover:bg-[#0E172B]">
                  <GitBranch size={15} /> {t('openTraceback')}
                </button>
              </div>
            </section>

            <section className="mb-6">
              <PipelinePanel result={result} language={language} />
            </section>

            <section className="mb-6 grid gap-6">
              <PriceRegimeChart data={result.time_series} stats={result.regime_stats} segments={result.regime_segments} />
              <RegimeConfidenceChart data={result.time_series} stats={result.regime_stats} language={language} />
              <RegimeSummaryMetrics result={result} language={language} />
            </section>

            <section className="mb-6 grid gap-6 xl:grid-cols-[.95fr_1.05fr]">
              <MemoPanel memo={result.memo} language={language} />
              <TransitionMatrix matrix={result.transition_matrix} labels={result.transition_labels} />
            </section>

            <section className="mb-6">
              <RegimeProbabilityStrip stats={result.regime_stats} currentRegime={result.current_regime} diagnostics={result.diagnostics} />
            </section>

            <section className="mb-6 grid gap-6 xl:grid-cols-3">
              <VolatilityChart data={result.time_series} />
              <DrawdownChart data={result.time_series} />
              <FeatureBehaviorChart data={result.time_series} />
            </section>
          </>
        )}

        {result && activeTab === 'traceback' && (
          <section className="mb-10">
            <RegimeTracebackPanel result={result} language={language} />
          </section>
        )}

        {result && activeTab === 'validation' && (
          <>
            <section className="mb-6 grid gap-6 xl:grid-cols-2">
              <BenchmarkPanel baseline={result.baseline} stability={result.stability} />
              <DataQualityPanel quality={result.data_quality} />
            </section>
            <section className="mb-6 grid gap-6 xl:grid-cols-[1fr_1fr]">
              <BaselineComparisonPanel baseline={result.baseline} />
              <TimelineSegments segments={result.regime_segments} />
            </section>
            <section className="mb-6 grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
              <DiagnosticsPanel diagnostics={result.diagnostics} />
              <RegimeStatsChart stats={result.regime_stats} language={language} />
            </section>
            <section className="mb-6">
              <ModelEvaluationPanel result={result} />
            </section>
            <section className="mb-6">
              <RegimeProfiles stats={result.regime_stats} t={t} language={language} />
            </section>
            <section className="mb-10">
              <ModelCardPanel modelCard={result.model_card} />
            </section>
          </>
        )}

        {activeTab === 'compare' && (
          <ComparePanel
            assets={assets}
            selectedAssets={selectedCompareAssets}
            setSelectedAssets={setSelectedCompareAssets}
            result={comparison}
            onRun={runComparison}
            loading={loading}
            language={language}
          />
        )}

        {activeTab === 'case' && <CaseStudyPanel caseStudy={caseStudy} projectCard={projectCard} />}

        {result && activeTab === 'export' && (
          <ExportPanel
            result={result}
            onDownloadReport={handleDownloadReport}
            onDownloadJson={handleDownloadJson}
            onCopySummary={copyPortfolioSummary}
          />
        )}
      </main>

      {notice && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-6 right-6 z-50 flex max-w-[calc(100vw-3rem)] items-center gap-3 rounded-2xl border border-emerald-200 bg-white px-4 py-3 text-sm font-semibold text-ink shadow-2xl shadow-black/15"
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-50 text-emerald-700">
            <CheckCircle2 size={18} />
          </span>
          {notice.message}
        </div>
      )}

      <footer className="border-t border-line bg-white/80 px-5 py-6 text-center text-xs text-subdued lg:px-8">
        <span className="font-semibold text-muted">QuantRegimeTracer</span> · Built by Oriol Martínez
      </footer>
    </div>
  )
}

function HeroStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs text-subdued">{label}</div>
      <div className="mt-2 text-xl font-semibold text-white">{value}</div>
    </div>
  )
}

function RegimeProfiles({ stats, t, language }: { stats?: RegimeStats[]; t: (key: string) => string; language: LanguageCode }) {
  return (
    <div className="card p-6">
      <p className="label">{t('regimeProfiles')}</p>
      <h3 className="mt-2 text-lg font-semibold text-ink">{t('stateStats')}</h3>
      <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {(stats || []).map((stat) => (
          <div key={stat.regime} className="rounded-2xl border border-line bg-white p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h4 className="font-semibold text-ink">{translateRegimeLabel(stat.label, language)}</h4>
              <span className="rounded-full bg-soft px-2 py-1 text-xs font-semibold text-brand">S{stat.regime}</span>
            </div>
            <div className="space-y-2 text-sm text-muted">
              <div className="flex justify-between"><span>{t('observations')}</span><strong>{stat.observations}</strong></div>
              <div className="flex justify-between"><span>{t('meanReturn')}</span><strong>{pct(stat.mean_return)}</strong></div>
              <div className="flex justify-between"><span>{t('annVolatility')}</span><strong>{pct(stat.annualized_volatility)}</strong></div>
              <div className="flex justify-between"><span>{t('meanDrawdown')}</span><strong>{pct(stat.mean_drawdown)}</strong></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
