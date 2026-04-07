"""
Hybrid Result Merger.

Merges YOLO physical metrics with Gemma 4 tactical analysis into a unified
output schema. Rules:
  - Physical metrics: YOLO only (precise positions)
  - Tactical metrics: YOLO compactness/width/depth + Gemma 4 formations/phases
  - Events: Gemma 4 only (or heuristic fallback)
  - Formation consensus: Compare YOLO vs Gemma 4 formation detection
  - Timeline: Chronological merge of all events and phases
"""
import time
from collections import defaultdict


class HybridResultMerger:
    """Merges YOLO and Gemma 4 analysis results."""

    def merge(
        self,
        yolo_metrics: dict,
        gemma4_results: dict,
        events: list,
        all_tracks: list,
        fps: float,
    ) -> dict:
        """Merge YOLO metrics and Gemma 4 results into unified schema.

        Args:
            yolo_metrics: Output from compute_metrics() — physical, tactical, individual
            gemma4_results: Output from ContextAwareGemma4Analyzer.analyze_key_frames()
            events: Output from Gemma4EventDetector.detect_and_classify()
            all_tracks: Full YOLO tracking data
            fps: Video FPS

        Returns:
            Unified hybrid_analysis.json schema
        """
        print("  Merging YOLO + Gemma 4 results...")

        total_frames = len(all_tracks)
        duration_s = total_frames / fps if fps > 0 else 0

        result = {
            'pipeline_version': 'hybrid_v1',
            'data_sources': {
                'yolo': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration_s': round(duration_s, 2),
                    'tracks_count': sum(len(ft) for ft in all_tracks),
                },
                'gemma4': {
                    'frames_analyzed': gemma4_results.get('summary', {}).get(
                        'frames_analyzed', 0
                    ),
                    'events_detected': len(events),
                },
            },
            # Physical metrics: YOLO only
            'physical': self._build_physical(yolo_metrics),
            # Individual metrics: YOLO only
            'individual': self._build_individual(yolo_metrics),
            # Tactical metrics: hybrid
            'tactical': self._build_tactical(yolo_metrics, gemma4_results),
            # Events: Gemma 4 (or heuristic fallback)
            'events': self._build_events(events),
            # Timeline: chronological merge
            'timeline': self._build_timeline(gemma4_results, events),
            # Summary
            'summary': yolo_metrics.get('summary', {}),
        }

        # Add pipeline metadata
        result['summary']['pipeline'] = 'hybrid_v1'
        result['summary']['gemma4_frames'] = gemma4_results.get(
            'summary', {}
        ).get('frames_analyzed', 0)
        result['summary']['events_count'] = len(events)

        print(f"  Merge complete: {len(result['physical'])} players, "
              f"{len(result['events'])} events, "
              f"{len(result['timeline'])} timeline entries")

        return result

    def _build_physical(self, yolo_metrics: dict) -> dict:
        """Physical metrics are YOLO-only (precise tracking positions)."""
        return yolo_metrics.get('physical', {})

    def _build_individual(self, yolo_metrics: dict) -> dict:
        """Individual metrics are YOLO-only."""
        individual = {}
        for pid, data in yolo_metrics.get('individual', {}).items():
            # Strip heatmap_data (too large for JSON output)
            entry = {k: v for k, v in data.items() if k != 'heatmap_data'}
            individual[pid] = entry
        return individual

    def _build_tactical(self, yolo_metrics: dict, gemma4_results: dict) -> dict:
        """Tactical metrics: YOLO spatial + Gemma 4 formations/phases."""
        yolo_tactical = yolo_metrics.get('tactical', {})
        gemma4_summary = gemma4_results.get('summary', {})
        gemma4_timeline = gemma4_results.get('tactical_timeline', [])

        tactical = {}

        # Per-team tactical data
        for team_key in ['team_0', 'team_1']:
            yolo_team = yolo_tactical.get(team_key, {})
            team_letter = 'a' if team_key == 'team_0' else 'b'

            # YOLO spatial metrics
            entry = {
                'yolo_formation': yolo_team.get('formation'),
                'yolo_formation_score': yolo_team.get('formation_score'),
                'compactness': yolo_team.get('compactness', {}),
                'width_depth': yolo_team.get('width_depth', {}),
                'n_players': yolo_team.get('n_players', 0),
            }

            # Gemma 4 formation (most common across analyzed frames)
            gemma4_formations = gemma4_summary.get(
                f'formations_team_{team_letter}', {}
            )
            if gemma4_formations:
                # Most frequent formation
                best_formation = max(gemma4_formations, key=gemma4_formations.get)
                entry['gemma4_formation'] = best_formation
                entry['gemma4_formation_votes'] = gemma4_formations
            else:
                entry['gemma4_formation'] = None

            # Formation consensus
            yf = entry['yolo_formation']
            gf = entry['gemma4_formation']
            if yf and gf:
                entry['formation_consensus'] = 'agreed' if yf == gf else 'disagreed'
                entry['formation_final'] = yf if yf == gf else gf  # Trust Gemma4 on disagreement
            elif yf:
                entry['formation_consensus'] = 'yolo_only'
                entry['formation_final'] = yf
            elif gf:
                entry['formation_consensus'] = 'gemma4_only'
                entry['formation_final'] = gf
            else:
                entry['formation_consensus'] = 'unknown'
                entry['formation_final'] = None

            tactical[team_key] = entry

        # Zone control (YOLO only)
        if 'zone_control' in yolo_tactical:
            tactical['zone_control'] = yolo_tactical['zone_control']

        # Gemma 4 tactical timeline
        tactical['gemma4_tactical_timeline'] = [
            {
                'timestamp_s': entry.get('timestamp_s', 0),
                'frame_index': entry.get('frame_index', 0),
                'phase_of_play': entry.get('phase_of_play'),
                'ball_possession': entry.get('ball_possession'),
                'pressing_intensity': entry.get('pressing_intensity'),
                'tactical_observation': entry.get('tactical_observation'),
                'key_patterns': entry.get('key_patterns', []),
            }
            for entry in gemma4_timeline
            if entry
        ]

        # Phase of play breakdown
        phases = gemma4_summary.get('phases_of_play', {})
        total_phases = sum(phases.values()) if phases else 0
        if total_phases > 0:
            tactical['phase_breakdown'] = {
                phase: {
                    'count': count,
                    'percentage': round(count / total_phases * 100, 1),
                }
                for phase, count in sorted(
                    phases.items(), key=lambda x: x[1], reverse=True
                )
            }

        return tactical

    def _build_events(self, events: list) -> list:
        """Build events list from Gemma 4 classifications."""
        return [
            {
                'timestamp_s': e.get('timestamp_s', 0),
                'frame_idx': e.get('frame_idx', 0),
                'event_type': e.get('event_type', 'other'),
                'confidence': e.get('confidence', 0),
                'description': e.get('description', ''),
                'involved_players': e.get('involved_players', []),
                'attacking_team': e.get('attacking_team'),
                'outcome': e.get('outcome'),
                'source': e.get('source', 'unknown'),
            }
            for e in events
        ]

    def _build_timeline(self, gemma4_results: dict, events: list) -> list:
        """Build chronological timeline from all sources."""
        timeline = []

        # Add tactical phases
        for entry in gemma4_results.get('tactical_timeline', []):
            if not entry:
                continue
            timeline.append({
                'timestamp_s': entry.get('timestamp_s', 0),
                'type': 'tactical',
                'phase_of_play': entry.get('phase_of_play'),
                'ball_possession': entry.get('ball_possession'),
                'observation': entry.get('tactical_observation', ''),
            })

        # Add events
        for event in events:
            timeline.append({
                'timestamp_s': event.get('timestamp_s', 0),
                'type': 'event',
                'event_type': event.get('event_type'),
                'description': event.get('description', ''),
                'confidence': event.get('confidence', 0),
            })

        # Sort chronologically
        timeline.sort(key=lambda x: x['timestamp_s'])

        return timeline
