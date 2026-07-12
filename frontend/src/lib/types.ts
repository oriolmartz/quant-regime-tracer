export type LanguageCode = 'en' | 'es'
export type RiskTone = 'red' | 'amber' | 'green' | 'blue' | 'neutral'
export type DataMode = 'real' | 'auto' | 'sample'


export interface BaselineDisagreementSegment {
  start: string
  end: string
  observations: number
  hmm_stress_share?: number
  baseline_stress_share?: number
}

export interface BaselineSuiteItem {
  name?: string
  description?: string
  latest_label?: string
  stress_agreement?: number
  disagreement_rate?: number
  verdict?: string
  threshold?: number | null
  latest_value?: number | null
  disagreement_segments?: BaselineDisagreementSegment[]
}

export interface BaselineReport {
  name?: string
  description?: string
  latest_state?: number
  latest_label?: string
  stress_agreement?: number
  disagreement_rate?: number
  verdict?: string
  suite_mean_agreement?: number
  suite_verdict?: string
  baseline_suite?: BaselineSuiteItem[]
  distribution?: Array<{ bucket: number; observations: number; share: number }>
  disagreement_segments?: BaselineDisagreementSegment[]
  interpretation?: string
}

export interface StabilityReport {
  status?: string
  message?: string
  frobenius_distance?: number | null
}

export interface DataQualityReport {
  status?: string
  source?: string
  cache_hit?: boolean
  observations?: number
  date_start?: string | null
  date_end?: string | null
  date_span_days?: number
  largest_gap_days?: number
  duplicate_dates?: number
  missing_close?: number
  missing_volume?: number | null
  annualization_factor?: number | null
  annualization_calendar?: string | null
  annualization_method?: string | null
  notes?: string[]
}

export interface DiagnosticsReport {
  model_type?: string
  n_observations?: number
  n_features?: number
  n_regimes?: number
  confidence_status?: string
  separability_score?: number | null
  notes?: string[]
  [key: string]: unknown
}

export interface RegimeSegment {
  start: string
  end: string
  regime: number
  label: string
  observations: number
  segment_return?: number
  max_drawdown?: number
}

export interface ModelCard {
  model_name?: string
  version?: string
  model_type?: string
  intended_use?: string
  not_intended_for?: string[]
  inputs?: string[]
  outputs?: string[]
  assumptions?: string[]
  failure_modes?: string[]
  guardrails?: string[]
  [key: string]: unknown
}

export interface RankedAssetSummary {
  asset: string
  rank?: number
  risk_score?: number
  current_label?: string
  current_regime?: string
  confidence?: number
  assignment_type?: string
  evidence_strength?: number
  risk_band?: string
  stress_transition_probability?: number
  stay_probability?: number
  latest_drawdown?: number
  annualized_volatility?: number
  annualization_factor?: number
  annualization_calendar?: string
  baseline_agreement?: number
  baseline_verdict?: string
  data_quality_status?: string
  source?: string
  [key: string]: unknown
}

export interface LanguageOption {
  code: LanguageCode
  label: string
  name: string
}


export interface SourceReport {
  mode?: DataMode | 'uploaded_csv' | 'custom'
  source?: string
  is_real_data?: boolean
  provider?: string
  cache_hit?: boolean
  requested_start?: string | null
  requested_end?: string | null
  actual_start?: string | null
  actual_end?: string | null
  observations?: number
  annualization_factor?: number | null
  annualization_calendar?: string | null
  annualization_method?: string | null
  policy?: string
}


export interface TracebackFeatureEvidence {
  feature: string
  value?: number | null
  percentile?: number | null
  signal?: string
  rationale?: string
}

export interface TracebackBaselineVote {
  name: string
  stress_vote?: boolean
  agrees_with_hmm_stress?: boolean
  latest_value?: number | null
  threshold?: number | null
  rule?: string
}

export interface TracebackStateProbability {
  state: number
  probability: number
}

export interface RegimeTracebackPoint {
  date: string
  close?: number | null
  assigned_state: number
  semantic_label: string
  previous_state?: number | null
  previous_label?: string | null
  posterior_confidence?: number | null
  posterior_entropy?: number | null
  assignment_type?: string
  evidence_strength?: number | null
  transition_prior?: number | null
  baseline_agreement_count?: number
  baseline_total?: number
  baseline_agreement_share?: number
  baseline_stress_votes?: number
  state_probabilities?: TracebackStateProbability[]
  feature_evidence?: TracebackFeatureEvidence[]
  baseline_votes?: TracebackBaselineVote[]
  event_tags?: string[]
  inference_path?: string[]
  interpretation?: string
}

export interface RegimeTracebackReport {
  name?: string
  summary?: string
  current?: RegimeTracebackPoint | null
  points?: RegimeTracebackPoint[]
  methodology?: string[]
}

export interface AnalyzeRequest {
  asset: string
  start?: string | null
  end?: string | null
  interval?: string
  n_regimes?: number
  data_mode?: DataMode
  prefer_live_data?: boolean
  force_refresh?: boolean
  language?: LanguageCode
}

export interface CompareRequest {
  assets: string[]
  start?: string | null
  end?: string | null
  interval?: string
  n_regimes?: number
  data_mode?: DataMode
  prefer_live_data?: boolean
  force_refresh?: boolean
  language?: LanguageCode
}

export interface RegimePoint {
  date: string
  close?: number
  price?: number
  return?: number
  log_return?: number
  rolling_volatility?: number
  drawdown?: number
  regime?: number | string
  regime_label?: string
  baseline_label?: string
  regime_probability?: number
  posterior_entropy?: number
  state_probability_0?: number
  state_probability_1?: number
  state_probability_2?: number
  state_probability_3?: number
  state_probability_4?: number
  [key: string]: string | number | boolean | null | undefined
}

export interface CurrentRegime {
  state?: number | string
  label: string
  confidence?: number
  risk_score?: number
  stay_probability?: number
  stress_transition_probability?: number
  expected_persistence_days?: number | null
  posterior_entropy?: number
  transition_entropy?: number
  assignment_strength?: number
  assignment_type?: string
  evidence_strength?: number
  evidence_components?: Record<string, number>
  confidence_interpretation?: string
  [key: string]: unknown
}

export interface RegimeStats {
  regime: number | string
  label: string
  observations?: number
  latest_probability?: number
  mean_return?: number
  annualized_volatility?: number
  mean_drawdown?: number
  mean_momentum?: number
  [key: string]: unknown
}

export interface MemoSectioned {
  summary?: string[]
  evidence?: string[]
  review?: string[]
  limitations?: string[]
  [key: string]: string[] | string | undefined
}

export interface AnalysisResult {
  api_version?: string
  asset: string
  start?: string
  end?: string
  source?: string
  current_regime: CurrentRegime
  time_series: RegimePoint[]
  regime_stats?: RegimeStats[]
  transition_matrix?: number[][]
  transition_labels?: string[]
  baseline?: BaselineReport
  stability?: StabilityReport
  data_quality?: DataQualityReport
  source_report?: SourceReport
  diagnostics?: DiagnosticsReport
  regime_segments?: RegimeSegment[]
  model_card?: ModelCard
  model_evaluation?: Record<string, unknown>
  traceback?: RegimeTracebackReport
  current_traceback?: RegimeTracebackPoint | null
  traceback_points?: RegimeTracebackPoint[]
  memo?: MemoSectioned | string | Record<string, unknown>
  report_markdown?: string
  warnings?: string[]
  [key: string]: unknown
}

export interface CompareResult {
  api_version?: string
  assets?: string[]
  ranked_assets?: RankedAssetSummary[]
  summaries?: RankedAssetSummary[]
  memo?: MemoSectioned | string | Record<string, unknown>
  portfolio_memo?: { summary?: string; review_notes?: string[]; [key: string]: unknown }
  highest_risk_asset?: string | null
  lowest_risk_asset?: string | null
  average_risk_score?: number
  [key: string]: unknown
}

export interface ProjectCard {
  name?: string
  tagline?: string
  description?: string
  stack?: string[]
  positioning?: string
  [key: string]: unknown
}

export interface CaseStudy {
  title?: string
  problem?: string
  architecture?: string[]
  approach?: string[]
  limitations?: string[]
  [key: string]: unknown
}
