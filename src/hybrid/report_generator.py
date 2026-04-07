"""
Hybrid HTML Report Generator.

Extends the standard YOLO report with:
- Match timeline (phase-of-play color bars, event markers)
- Tactic comparison (YOLO vs Gemma 4 formations)
- Phase-of-play breakdown (pie chart)
- Event list (chronological, with Gemma 4 descriptions)
- AI Tactical Insights (Gemma 4 observations)
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np


class HybridReportGenerator:
    """Generate enhanced HTML report from hybrid analysis results."""

    # Phase colors for timeline visualization
    PHASE_COLORS = {
        'attack': '#4CAF50',
        'defense': '#F44336',
        'transition_attack': '#8BC34A',
        'transition_defense': '#FF9800',
        'build_up': '#2196F3',
        'pressing': '#FF5722',
        'set_piece': '#9C27B0',
        'goal_kick': '#607D8B',
        'corner': '#795548',
        'free_kick': '#FFC107',
        'throw_in': '#00BCD4',
    }

    EVENT_ICONS = {
        'goal_attempt': '&#9917;',
        'goal': '&#9917;&#127881;',
        'pass': '&#10132;',
        'cross': '&#10548;',
        'tackle': '&#9889;',
        'foul': '&#128308;',
        'corner': '&#128204;',
        'free_kick': '&#128312;',
        'interception': '&#128737;',
        'substitution': '&#128260;',
        'other': '&#8226;',
    }

    def generate(
        self,
        merged_results: dict,
        yolo_metrics: dict,
        output_dir: str,
    ) -> str:
        """Generate enhanced HTML report.

        Args:
            merged_results: Output from HybridResultMerger.merge()
            yolo_metrics: Raw YOLO metrics (for heatmaps, detailed charts)
            output_dir: Directory to save report

        Returns:
            Path to generated HTML file.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        phys = merged_results.get('physical', {})
        indiv = merged_results.get('individual', {})
        tactical = merged_results.get('tactical', {})
        events = merged_results.get('events', [])
        timeline = merged_results.get('timeline', [])
        summary = merged_results.get('summary', {})

        # Sort players by distance
        sorted_players = sorted(
            phys.items(),
            key=lambda x: x[1].get('total_distance_km', 0),
            reverse=True
        )
        player_ids = [p[0] for p in sorted_players]

        # Generate heatmap images
        heatmap_imgs = self._generate_heatmaps(player_ids[:6], yolo_metrics, plt)

        # Build chart data
        distances = [phys[p].get('total_distance_km', 0) for p in player_ids]
        avg_speeds = [phys[p].get('avg_speed_kmh', 0) for p in player_ids]
        max_speeds = [phys[p].get('max_speed_kmh', 0) for p in player_ids]
        teams = [indiv.get(p, {}).get('team', -1) for p in player_ids]
        team_colors = [
            '#0064FF' if t == 0 else '#FF0000' if t == 1 else '#999'
            for t in teams
        ]
        labels = [f"#{p}" for p in player_ids]

        # Speed zone data
        zone_names = ['walking', 'jogging', 'running', 'high_speed', 'sprinting']
        zone_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        zone_data = {zn: [] for zn in zone_names}
        for p in player_ids:
            for zn in zone_names:
                zone_data[zn].append(
                    phys[p].get('speed_zones', {}).get(zn, {}).get('percentage', 0)
                )

        # Build HTML sections
        overview_html = self._build_overview(summary)
        formation_html = self._build_formation_comparison(tactical)
        timeline_html = self._build_timeline_section(timeline, summary)
        events_html = self._build_events_section(events)
        phase_breakdown_html = self._build_phase_breakdown(tactical)
        insights_html = self._build_ai_insights(tactical)
        heatmap_html = self._build_heatmap_section(heatmap_imgs)
        summary_table_html = self._build_summary_table(player_ids, phys, indiv)

        # Zone control
        zc = tactical.get('zone_control', {'team_a': 50, 'team_b': 50})

        # Speed zone traces for Plotly
        zone_traces = []
        for i, zn in enumerate(zone_names):
            zone_traces.append(
                f'{{"x": {json.dumps(labels)}, "y": {json.dumps(zone_data[zn])}, '
                f'"name": "{zn.replace("_", " ").title()}", "type": "bar", '
                f'"marker": {{"color": "{zone_colors[i]}"}}}}'
            )
        zone_traces_js = ",\n    ".join(zone_traces)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hybrid Football Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1400px; margin: auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 30px; }}
    h3 {{ color: #3949ab; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
             box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
    .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
    .chart {{ min-height: 400px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background: #e8eaf6; font-weight: 600; }}
    tr:hover {{ background: #f5f5f5; }}
    .summary-stat {{ display: inline-block; background: #e8eaf6; padding: 10px 20px;
                     border-radius: 8px; margin: 5px; text-align: center; }}
    .summary-stat .value {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
    .summary-stat .label {{ font-size: 12px; color: #666; }}
    .heatmaps {{ display: flex; flex-wrap: wrap; justify-content: center; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
              font-size: 12px; font-weight: 600; color: white; margin: 2px; }}
    .badge-green {{ background: #4CAF50; }}
    .badge-red {{ background: #F44336; }}
    .badge-orange {{ background: #FF9800; }}
    .badge-blue {{ background: #2196F3; }}
    .badge-purple {{ background: #9C27B0; }}
    .formation-box {{ display: inline-block; padding: 8px 16px; border-radius: 8px;
                      font-size: 20px; font-weight: bold; margin: 5px;
                      border: 2px solid #ddd; }}
    .formation-agreed {{ border-color: #4CAF50; background: #E8F5E9; }}
    .formation-disagreed {{ border-color: #FF9800; background: #FFF3E0; }}
    .timeline-bar {{ display: flex; height: 30px; border-radius: 4px; overflow: hidden;
                     margin: 10px 0; }}
    .timeline-segment {{ height: 100%; transition: opacity 0.2s; cursor: pointer; }}
    .timeline-segment:hover {{ opacity: 0.8; }}
    .event-card {{ border-left: 4px solid #1a237e; padding: 8px 16px; margin: 8px 0;
                   background: #fafafa; border-radius: 0 4px 4px 0; }}
    .event-time {{ font-weight: bold; color: #1a237e; }}
    .insight-card {{ border-left: 4px solid #9C27B0; padding: 12px 16px; margin: 8px 0;
                     background: #F3E5F5; border-radius: 0 4px 4px 0; }}
    .hybrid-badge {{ background: linear-gradient(135deg, #1a237e, #9C27B0);
                     color: white; padding: 4px 12px; border-radius: 12px;
                     font-size: 11px; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
<h1>Hybrid Football Analysis Report
    <span class="hybrid-badge">YOLO + Gemma 4</span>
</h1>

{overview_html}

{formation_html}

{timeline_html}

{phase_breakdown_html}

{events_html}

{insights_html}

<div class="grid">
    <div class="card">
        <h2>Distance Covered (km)</h2>
        <div id="distance-chart" class="chart"></div>
    </div>
    <div class="card">
        <h2>Speed (km/h)</h2>
        <div id="speed-chart" class="chart"></div>
    </div>
</div>

<div class="card">
    <h2>Speed Zone Distribution (%)</h2>
    <div id="zone-chart" class="chart"></div>
</div>

<div class="grid">
    <div class="card">
        <h2>Zone Control</h2>
        <div id="zone-control-chart" class="chart"></div>
    </div>
    <div class="card">
        <h2>Phase of Play Distribution</h2>
        <div id="phase-chart" class="chart"></div>
    </div>
</div>

{heatmap_html}

{summary_table_html}

</div>

<script>
// Distance chart
Plotly.newPlot('distance-chart', [{{
    x: {json.dumps(labels)},
    y: {json.dumps(distances)},
    type: 'bar',
    marker: {{ color: {json.dumps(team_colors)} }}
}}], {{
    margin: {{t: 10, b: 60}},
    yaxis: {{title: 'km'}},
    xaxis: {{tickangle: -45}}
}});

// Speed chart
Plotly.newPlot('speed-chart', [
    {{ x: {json.dumps(labels)}, y: {json.dumps(avg_speeds)}, name: 'Avg', type: 'bar',
       marker: {{color: '#42A5F5'}} }},
    {{ x: {json.dumps(labels)}, y: {json.dumps(max_speeds)}, name: 'Max', type: 'bar',
       marker: {{color: '#EF5350'}} }}
], {{
    barmode: 'group',
    margin: {{t: 10, b: 60}},
    yaxis: {{title: 'km/h'}},
    xaxis: {{tickangle: -45}}
}});

// Speed zone stacked bar
Plotly.newPlot('zone-chart', [
    {zone_traces_js}
], {{
    barmode: 'stack',
    margin: {{t: 10, b: 60}},
    yaxis: {{title: '%'}},
    xaxis: {{tickangle: -45}},
    legend: {{orientation: 'h', y: 1.1}}
}});

// Zone control pie
Plotly.newPlot('zone-control-chart', [{{
    values: [{zc.get('team_a', 50):.1f}, {zc.get('team_b', 50):.1f}],
    labels: ['Team A', 'Team B'],
    type: 'pie',
    marker: {{ colors: ['#0064FF', '#FF0000'] }},
    hole: 0.4
}}], {{
    margin: {{t: 30, b: 30}}
}});

// Phase of play pie
{self._build_phase_chart_js(tactical)}
</script>
</body>
</html>"""

        report_path = os.path.join(output_dir, "hybrid_report.html")
        with open(report_path, 'w') as f:
            f.write(html)

        return report_path

    def _build_overview(self, summary: dict) -> str:
        n_players = summary.get('n_players_tracked', 0)
        duration = summary.get('duration_seconds', 0)
        total_frames = summary.get('total_frames', 0)
        n_clips = summary.get('n_clips', 0)
        fps = summary.get('fps', 30)
        gemma4_frames = summary.get('gemma4_frames', 0)
        events_count = summary.get('events_count', 0)

        return f"""
<div class="card">
    <h2>Overview</h2>
    <div style="text-align: center;">
        <div class="summary-stat">
            <div class="value">{n_players}</div>
            <div class="label">Players Tracked</div>
        </div>
        <div class="summary-stat">
            <div class="value">{duration:.0f}s</div>
            <div class="label">Duration</div>
        </div>
        <div class="summary-stat">
            <div class="value">{total_frames}</div>
            <div class="label">Total Frames</div>
        </div>
        <div class="summary-stat">
            <div class="value">{gemma4_frames}</div>
            <div class="label">AI-Analyzed Frames</div>
        </div>
        <div class="summary-stat">
            <div class="value">{events_count}</div>
            <div class="label">Events Detected</div>
        </div>
        <div class="summary-stat">
            <div class="value">{fps:.0f}</div>
            <div class="label">FPS</div>
        </div>
    </div>
</div>"""

    def _build_formation_comparison(self, tactical: dict) -> str:
        rows = ""
        for team_key in ['team_0', 'team_1']:
            team = tactical.get(team_key, {})
            team_letter = 'A' if team_key == 'team_0' else 'B'
            yf = team.get('yolo_formation', '—')
            gf = team.get('gemma4_formation', '—')
            consensus = team.get('formation_consensus', 'unknown')
            final = team.get('formation_final', '—')
            css = 'formation-agreed' if consensus == 'agreed' else 'formation-disagreed'

            comp = team.get('compactness', {})
            wd = team.get('width_depth', {})

            rows += f"""
            <tr>
                <td><strong>Team {team_letter}</strong></td>
                <td><span class="formation-box">{yf or '—'}</span></td>
                <td><span class="formation-box">{gf or '—'}</span></td>
                <td><span class="formation-box {css}">{final or '—'}</span>
                    <span class="badge {'badge-green' if consensus == 'agreed' else 'badge-orange'}">{consensus}</span>
                </td>
                <td>{comp.get('area_m2', 0):.0f} m&sup2;</td>
                <td>{wd.get('width_m', 0):.1f}m x {wd.get('depth_m', 0):.1f}m</td>
            </tr>"""

        return f"""
<div class="card">
    <h2>Formation Analysis (YOLO vs Gemma 4)</h2>
    <table>
        <thead><tr>
            <th>Team</th><th>YOLO Formation</th><th>Gemma 4 Formation</th>
            <th>Consensus</th><th>Compactness</th><th>Width x Depth</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""

    def _build_timeline_section(self, timeline: list, summary: dict) -> str:
        if not timeline:
            return ""

        duration = summary.get('duration_seconds', 1)
        segments = ""
        for entry in timeline:
            if entry.get('type') != 'tactical':
                continue
            phase = entry.get('phase_of_play', 'other')
            color = self.PHASE_COLORS.get(phase, '#999')
            pct = max(1, 100 / max(1, len([e for e in timeline if e.get('type') == 'tactical'])))
            tooltip = f"{entry.get('timestamp_s', 0):.1f}s: {phase}"
            if entry.get('ball_possession'):
                tooltip += f" (possession: {entry['ball_possession']})"
            segments += (
                f'<div class="timeline-segment" '
                f'style="width:{pct:.1f}%;background:{color};" '
                f'title="{tooltip}"></div>'
            )

        # Event markers
        event_markers = ""
        for entry in timeline:
            if entry.get('type') != 'event':
                continue
            ts = entry.get('timestamp_s', 0)
            pct = (ts / duration * 100) if duration > 0 else 0
            icon = self.EVENT_ICONS.get(entry.get('event_type', 'other'), '&#8226;')
            event_markers += (
                f'<div style="position:absolute;left:{pct:.1f}%;top:-20px;'
                f'font-size:16px;cursor:pointer;" '
                f'title="{ts:.1f}s: {entry.get("description", "")}">{icon}</div>'
            )

        # Legend
        legend = " ".join(
            f'<span class="badge" style="background:{color}">{phase}</span>'
            for phase, color in self.PHASE_COLORS.items()
        )

        return f"""
<div class="card">
    <h2>Match Timeline</h2>
    <div style="position:relative;padding-top:25px;">
        {event_markers}
        <div class="timeline-bar">{segments}</div>
    </div>
    <div style="margin-top:10px;font-size:12px;">{legend}</div>
</div>"""

    def _build_events_section(self, events: list) -> str:
        if not events:
            return ""

        event_cards = ""
        for e in sorted(events, key=lambda x: x.get('timestamp_s', 0)):
            icon = self.EVENT_ICONS.get(e.get('event_type', 'other'), '&#8226;')
            source_badge = (
                '<span class="badge badge-purple">AI</span>'
                if e.get('source') == 'gemma4'
                else '<span class="badge badge-blue">Heuristic</span>'
            )
            players = ", ".join(
                f"#{p}" for p in e.get('involved_players', [])
            )
            event_cards += f"""
            <div class="event-card">
                <span class="event-time">{e.get('timestamp_s', 0):.1f}s</span>
                {icon} <strong>{e.get('event_type', 'unknown').replace('_', ' ').title()}</strong>
                {source_badge}
                {f' — Players: {players}' if players else ''}
                <br><small>{e.get('description', '')}</small>
            </div>"""

        return f"""
<div class="card">
    <h2>Events ({len(events)})</h2>
    {event_cards}
</div>"""

    def _build_phase_breakdown(self, tactical: dict) -> str:
        breakdown = tactical.get('phase_breakdown', {})
        if not breakdown:
            return ""

        rows = ""
        for phase, data in breakdown.items():
            color = self.PHASE_COLORS.get(phase, '#999')
            rows += (
                f'<tr><td><span class="badge" style="background:{color}">'
                f'{phase.replace("_", " ").title()}</span></td>'
                f'<td>{data["count"]}</td>'
                f'<td>{data["percentage"]:.1f}%</td>'
                f'<td><div style="background:{color};height:12px;'
                f'width:{data["percentage"]}%;border-radius:2px;"></div></td></tr>'
            )

        return f"""
<div class="card">
    <h2>Phase of Play Breakdown</h2>
    <table>
        <thead><tr><th>Phase</th><th>Count</th><th>%</th><th></th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""

    def _build_ai_insights(self, tactical: dict) -> str:
        timeline_entries = tactical.get('gemma4_tactical_timeline', [])
        if not timeline_entries:
            return ""

        insights = ""
        for entry in timeline_entries:
            obs = entry.get('tactical_observation', '')
            patterns = entry.get('key_patterns', [])
            if not obs:
                continue
            patterns_html = ""
            if patterns:
                patterns_html = "<br><small>Patterns: " + ", ".join(patterns) + "</small>"
            insights += f"""
            <div class="insight-card">
                <span class="event-time">{entry.get('timestamp_s', 0):.1f}s</span>
                — <strong>{entry.get('phase_of_play', '').replace('_', ' ').title()}</strong>
                {f" (Possession: {entry['ball_possession']})" if entry.get('ball_possession') else ""}
                <br>{obs}
                {patterns_html}
            </div>"""

        return f"""
<div class="card">
    <h2>AI Tactical Insights <span class="hybrid-badge">Gemma 4</span></h2>
    {insights}
</div>"""

    def _build_heatmap_section(self, heatmap_imgs: dict) -> str:
        if not heatmap_imgs:
            return ""
        imgs = ""
        for pid, img_b64 in heatmap_imgs.items():
            imgs += (
                f'<img src="data:image/png;base64,{img_b64}" '
                f'style="margin:5px; border:1px solid #ddd; border-radius:4px;">\n'
            )
        return f"""
<div class="card">
    <h2>Player Heatmaps (Top Players)</h2>
    <div class="heatmaps">{imgs}</div>
</div>"""

    def _build_summary_table(self, player_ids, phys, indiv):
        rows = ""
        for pid in player_ids:
            p = phys.get(pid, {})
            i = indiv.get(pid, {})
            team_label = 'A' if i.get('team') == 0 else 'B' if i.get('team') == 1 else '?'
            role = i.get('role', 'player')
            role_label = {'goalkeeper': 'GK', 'player': ''}.get(role, role)
            avg_pos = i.get('average_position', {})
            pt = i.get('playing_time', {})
            rows += (
                f"<tr>"
                f"<td>#{pid}</td><td>{team_label}</td><td>{role_label}</td>"
                f"<td>{p.get('total_distance_km', 0):.3f}</td>"
                f"<td>{p.get('avg_speed_kmh', 0):.1f}</td>"
                f"<td>{p.get('max_speed_kmh', 0):.1f}</td>"
                f"<td>{p.get('sprint_count', 0)}</td>"
                f"<td>{pt.get('playing_time_minutes', 0):.1f}</td>"
                f"<td>({avg_pos.get('x', 0):.1f}, {avg_pos.get('y', 0):.1f})</td>"
                f"</tr>\n"
            )

        return f"""
<div class="card">
    <h2>Player Summary Table</h2>
    <table>
        <thead><tr>
            <th>Player</th><th>Team</th><th>Role</th><th>Distance (km)</th>
            <th>Avg Speed (km/h)</th><th>Max Speed (km/h)</th><th>Sprints</th>
            <th>Playing Time (min)</th><th>Avg Position</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""

    def _generate_heatmaps(self, player_ids, yolo_metrics, plt):
        """Generate heatmap images as base64."""
        heatmap_imgs = {}
        indiv = yolo_metrics.get('individual', {})
        for pid in player_ids:
            if pid in indiv and 'heatmap_data' in indiv[pid]:
                fig, ax = plt.subplots(1, 1, figsize=(5.25, 3.4))
                hm = np.array(indiv[pid]['heatmap_data'])
                ax.imshow(hm, cmap='hot', interpolation='bilinear', aspect='auto',
                          extent=[0, 105, 68, 0])
                team_label = f"Team {'A' if indiv[pid].get('team') == 0 else 'B'}"
                ax.set_title(f"Player #{pid} ({team_label})", fontsize=10)
                ax.set_xlabel("Length (m)")
                ax.set_ylabel("Width (m)")
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                heatmap_imgs[pid] = base64.b64encode(buf.read()).decode('utf-8')
        return heatmap_imgs

    def _build_phase_chart_js(self, tactical):
        """Build Plotly JS for phase of play pie chart."""
        breakdown = tactical.get('phase_breakdown', {})
        if not breakdown:
            return "// No phase data"

        labels = list(breakdown.keys())
        values = [breakdown[p]['count'] for p in labels]
        colors = [self.PHASE_COLORS.get(p, '#999') for p in labels]
        labels = [p.replace('_', ' ').title() for p in labels]

        return f"""
Plotly.newPlot('phase-chart', [{{
    values: {json.dumps(values)},
    labels: {json.dumps(labels)},
    type: 'pie',
    marker: {{ colors: {json.dumps(colors)} }},
    hole: 0.4
}}], {{
    margin: {{t: 30, b: 30}}
}});"""
