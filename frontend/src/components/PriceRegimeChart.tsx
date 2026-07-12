import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { RegimePoint, RegimeSegment, RegimeStats } from '../lib/types'

const palette = ['#D7E8E4', '#F3D7A7', '#E7B8B8', '#D6E2F0', '#E7DCCB']
const strokePalette = ['#2A6F68', '#B7791F', '#A04444', '#345C7D', '#765E3D']

function pct(value?: number | null, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

function buildSegments(data?: RegimePoint[]) {
  if (!data?.length) return []
  const segments = []
  let start = data[0].date
  let regime = data[0].regime
  let label = data[0].regime_label
  for (let i = 1; i < data.length; i++) {
    if (data[i].regime !== regime) {
      segments.push({ start, end: data[i - 1].date, regime, label })
      start = data[i].date
      regime = data[i].regime
      label = data[i].regime_label
    }
  }
  segments.push({ start, end: data[data.length - 1].date, regime, label })
  return segments
}

function regimeIndex(regime: string | number | undefined) {
  const parsed = Number(regime)
  return Number.isFinite(parsed) ? parsed : 0
}

function findSegment(pointDate: string, segments?: RegimeSegment[]) {
  return (segments || []).find((segment) => segment.start <= pointDate && pointDate <= segment.end)
}

function CustomTooltip({ active, payload, label, segments }: { active?: boolean; payload?: any[]; label?: string; segments?: RegimeSegment[] }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload as RegimePoint
  const segment = label ? findSegment(label, segments) : undefined
  return (
    <div className="rounded-2xl border border-line bg-white/95 p-4 text-xs shadow-premium backdrop-blur">
      <div className="mb-2 font-semibold text-ink">{label}</div>
      <div className="space-y-1 text-muted">
        <div><span className="font-medium text-ink">Adjusted close:</span>{' '}{Number(point.close ?? point.price ?? 0).toFixed(2)}</div>
        <div><span className="font-medium text-ink">State:</span> {point.regime} · {point.regime_label}</div>
        <div><span className="font-medium text-ink">MAP posterior γ:</span> {pct(Number(point.regime_probability ?? 0))}</div>
        <div><span className="font-medium text-ink">Entropy H(γ):</span> {pct(Number(point.posterior_entropy ?? 0))}</div>
        <div><span className="font-medium text-ink">Log return:</span> {pct(Number(point.log_return ?? 0), 2)}</div>
        <div><span className="font-medium text-ink">Volatility:</span> {pct(Number(point.rolling_volatility ?? 0), 1)}</div>
        <div><span className="font-medium text-ink">Drawdown:</span> {pct(Number(point.drawdown ?? 0), 1)}</div>
      </div>
      {segment && (
        <div className="mt-3 rounded-xl bg-ivory px-3 py-2 text-muted">
          <div><span className="font-medium text-ink">Segment:</span> {segment.start} → {segment.end}</div>
          <div><span className="font-medium text-ink">Duration:</span> {segment.observations} observations</div>
          <div><span className="font-medium text-ink">Segment return:</span> {pct(segment.segment_return, 1)}</div>
          <div><span className="font-medium text-ink">Max drawdown:</span> {pct(segment.max_drawdown, 1)}</div>
        </div>
      )}
    </div>
  )
}

export default function PriceRegimeChart({
  data = [],
  stats = [],
  segments: externalSegments = [],
}: {
  data?: RegimePoint[]
  stats?: RegimeStats[]
  segments?: RegimeSegment[]
}) {
  const segments = buildSegments(data)
  const transitions = segments.slice(1).map((segment) => segment.start)
  const regimes = [...stats].sort((a, b) => Number(a.regime) - Number(b.regime))

  return (
    <div className="card p-6">
      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="label">Regime chart</p>
          <h3 className="mt-2 text-lg font-semibold text-ink">Price path with inferred regimes</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
            The price path is overlaid with stronger regime bands and transition markers. Hover a point to inspect state ID, semantic label, MAP posterior mass, entropy, segment duration, return, volatility and drawdown.
          </p>
        </div>
        <div className="flex max-w-xl flex-wrap gap-2">
          {regimes.map((stat, idx) => (
            <span
              key={stat.regime}
              className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-3 py-1 text-xs font-semibold text-ink"
            >
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: strokePalette[idx % strokePalette.length] }} />
              State {stat.regime} · {stat.label}
            </span>
          ))}
        </div>
      </div>
      <div className="h-[430px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 14, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#E4DDD2" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#6B7280' }} tickMargin={10} minTickGap={36} />
            <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} domain={['auto', 'auto']} width={58} />
            <Tooltip content={<CustomTooltip segments={externalSegments} />} />
            {segments.map((seg, idx) => {
              const ri = regimeIndex(seg.regime)
              return (
                <ReferenceArea
                  key={`${seg.start}-${idx}`}
                  x1={seg.start}
                  x2={seg.end}
                  fill={palette[ri % palette.length]}
                  fillOpacity={0.44}
                  strokeOpacity={0}
                />
              )
            })}
            {transitions.map((date) => (
              <ReferenceLine key={date} x={date} stroke="#14213D" strokeDasharray="3 5" strokeOpacity={0.35} />
            ))}
            <Line
              type="monotone"
              dataKey="close"
              stroke="#14213D"
              strokeWidth={2.4}
              dot={false}
              name="Adjusted close"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted">
        <span className="font-semibold text-ink">Transition markers</span>
        <span>Dashed vertical lines mark latent-state changes.</span>
        <span>Background color is consistent with the state legend.</span>
      </div>
    </div>
  )
}
