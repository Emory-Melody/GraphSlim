#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Recalculate the frozen cohort reconciliation using only Python's standard library."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

METRICS = ('mean_shortfall_percent', 'mean_relative_gap_percent', 'mean_absolute_gap_pp')
THRESHOLDS = (1, 2, 5, 10)


def errors(reference, reproduced):
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(reproduced):
        raise ValueError('Scores must be finite and references must be positive')
    return (100 * max(0, reference - reproduced) / reference,
            100 * abs(reference - reproduced) / reference,
            abs(reference - reproduced))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(pairs, policy=False):
    return {metric: {str(t): dict(
        count=sum((policy and p['credited_by_policy']) or
                  (p['complete_result'] and p[metric] is not None and p[metric] < t)
                  for p in pairs), denominator=len(pairs)) for t in THRESHOLDS}
        for metric in METRICS}


def calculate(base):
    cohorts = load(base / 'cohorts.json')
    old, numeric, paper = [load(base / cohorts[k]) for k in ('historical', 'numerical_only', 'paper')]
    rows = load(base / 'settings.json')
    grouped = {}
    for row in rows:
        key = (row['method'], row['dataset'])
        grouped.setdefault(key, []).append(row)
        source = base / row['extraction_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['extraction_sha256'], source
    assert len(grouped) == len(paper['pairs']) == 124
    assert len(rows) == 355
    derived = []
    for pair in paper['pairs']:
        key = (pair['method'], pair['dataset'])
        settings = grouped[key]
        assert len(settings) == len({r['setting'] for r in settings})
        assert {r['setting'] for r in settings} == set(pair['declared_rates'])
        assert pair['reference_status'] == 'numeric_reference' and not pair['credited_by_policy']
        assert all(r['score_metric'] == settings[0]['score_metric'] for r in settings)
        values = [errors(r['reference_percent'], r['reproduced_percent']) for r in settings]
        result = dict(method=key[0], dataset=key[1], complete_result=pair['complete_result'],
                      credited_by_policy=False, settings=len(settings),
                      reference_source=pair['reference_source'],
                      shared_observations_with_reference=all(r['shared_observations_with_reference'] for r in settings))
        for i, metric in enumerate(METRICS):
            result[metric] = statistics.mean(v[i] for v in values)
            assert math.isclose(result[metric], pair[metric], rel_tol=1e-10, abs_tol=1e-10), (key, metric)
        derived.append(result)
    assert len({p['method'] for p in derived}) == paper['method_count'] == 19
    assert len({p['dataset'] for p in derived}) == 7
    assert sum(p['shared_observations_with_reference'] for p in derived) == 8
    assert old['pairs'] and len(old['pairs']) == 123 and old['method_count'] == 20
    assert [p for p in old['pairs'] if not p['credited_by_policy']] == numeric['pairs']
    assert paper['pairs'][:116] == numeric['pairs']
    assert all(p['method'] == 'clustering' and all(p[m] is None for m in METRICS)
               for p in old['pairs'] if p['credited_by_policy'])
    assert sum(p['credited_by_policy'] for p in old['pairs']) == 7
    # In the paper snapshot all scores are complete; completion never replaces a missing reference.
    views = dict(historical_with_policy=aggregate(old['pairs'], policy=True),
                 numerical_only_116=aggregate(numeric['pairs']),
                 paper_124=aggregate(derived))
    for name, snapshot in [('numerical_only_116', numeric), ('paper_124', paper)]:
        for metric in METRICS:
            for t, expected in snapshot['thresholds'][metric].items():
                assert views[name][metric][t]['count'] == expected['count']
    summary = dict(artifact_id='graphslim-reproducibility-20261103', setting_count=len(rows),
                   pair_count=len(derived), method_count=19, dataset_count=7,
                   threshold_rule='strict < after averaging per-setting errors within a pair',
                   shared_observation_pairs=8, cohort_results=views)
    uncorrected = []
    for pair in derived:
        settings = grouped[pair['method'], pair['dataset']]
        values = [errors((r['reference_correction']['table9'] if r['reference_correction']
                          else r['reference_percent']), r['reproduced_percent']) for r in settings]
        uncorrected.append(dict(pair, **{metric: statistics.mean(v[i] for v in values)
                                        for i, metric in enumerate(METRICS)}))
    summary['uncorrected_table9_sensitivity'] = aggregate(uncorrected)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(derived[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(derived)
    return summary, stream.getvalue()


def verify(base):
    manifest = load(base / 'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name
    summary, pairs = calculate(base)
    assert summary == load(base / 'summary.json'), 'Summary differs from setting-level recalculation'
    assert pairs == (base / 'pairs.csv').read_text(), 'Pair-level metrics differ'
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify file hashes and recompute every metric (default).')
    parser.add_argument('--write', action='store_true', help='Regenerate derived tables and manifest from fixed inputs.')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if args.write:
        summary, pairs = calculate(base)
        (base / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        (base / 'pairs.csv').write_text(pairs)
        manifest = dict(artifact_id=summary['artifact_id'], files_sha256={
            str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob('*')) if p.is_file() and p.name != 'manifest.json'
            and '__pycache__' not in p.parts})
        (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = verify(base)
    print(f"Verified {summary['setting_count']} settings, 124 pairs, 19 methods and 7 datasets.")
    for cohort, results in summary['cohort_results'].items():
        for metric in METRICS[:2]:
            values = ', '.join(f"<{t}% {v['count']}/{v['denominator']} ({100*v['count']/v['denominator']:.2f}%)"
                               for t, v in results[metric].items())
            print(f'{cohort} / {metric}: {values}')


if __name__ == '__main__':
    main()
