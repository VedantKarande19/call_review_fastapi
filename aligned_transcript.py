import re
from dataclasses import dataclass
from typing import Any, Iterable

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def dist_midpoint_to_interval(mid: float, lo: float, hi: float) -> float:
    if mid < lo: return lo - mid
    if mid > hi: return mid - hi
    return 0.0

def fmt_mm_ss(t: float) -> str:
    sec = max(0.0, t)
    m = int(sec // 60)
    s = int(round(sec % 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.,!?])", r"\1", s)
    return s

@dataclass
class DiarTurn:
    start: float
    end: float
    speaker: str
    index: int

@dataclass
class Word:
    start: float
    end: float
    text: str

def load_words(data: dict[str, Any]) -> list[Word]:
    raw = data.get("wordLevelTranscription") or []
    out: list[Word] = []
    for w in raw:
        t = str(w.get("text") or w.get("word") or "").strip()
        if not t: continue
        out.append(Word(start=float(w["start"]), end=float(w["end"]), text=t))
    out.sort(key=lambda x: x.start)
    return out

def load_diar(data: dict[str, Any]) -> list[DiarTurn]:
    raw = data.get("diarization") or []
    turns: list[DiarTurn] = []
    for i, d in enumerate(raw):
        turns.append(DiarTurn(start=float(d["start"]), end=float(d["end"]), speaker=str(d["speaker"]), index=i))
    turns.sort(key=lambda x: x.start)
    return turns

def assign_turn_index(w: Word, turns: list[DiarTurn]) -> int:
    best_i = 0
    best_o = -1.0
    for t in turns:
        o = overlap(w.start, w.end, t.start, t.end)
        if o > best_o:
            best_o = o
            best_i = t.index
    if best_o > 0:
        return best_i
    mid = 0.5 * (w.start + w.end)
    best_d = float("inf")
    for t in turns:
        d = dist_midpoint_to_interval(mid, t.start, t.end)
        if d < best_d:
            best_d = d
            best_i = t.index
    return best_i

EXPECTED_PARTY_COUNT = 2

def unique_speakers_by_duration(turns: list[DiarTurn]) -> list[tuple[str, float]]:
    dur: dict[str, float] = {}
    for t in turns:
        dur[t.speaker] = dur.get(t.speaker, 0.0) + max(0.0, t.end - t.start)
    return sorted(dur.items(), key=lambda x: -x[1])

def speaker_labels_two_party(turns: list[DiarTurn]) -> dict[str, str]:
    ranked = unique_speakers_by_duration(turns)
    if not ranked: return {}
    agent_spk = ranked[0][0]
    customer_spk = ranked[1][0] if len(ranked) > 1 else agent_spk
    mapping: dict[str, str] = {agent_spk: "Agent", customer_spk: "Customer"}
    for spk, _ in ranked[2:]:
        mapping[spk] = "Customer"
    for t in turns:
        mapping.setdefault(t.speaker, "Customer")
    return mapping

def build_rows(words: list[Word], turns: list[DiarTurn], label_by_speaker: dict[str, str]) -> list[dict[str, Any]]:
    by_index: dict[int, list[Word]] = {}
    for w in words:
        idx = assign_turn_index(w, turns)
        by_index.setdefault(idx, []).append(w)

    turn_by_index = {t.index: t for t in turns}
    rows: list[dict[str, Any]] = []
    for idx in sorted(by_index.keys(), key=lambda i: turn_by_index[i].start):
        bucket = by_index[idx]
        bucket.sort(key=lambda x: x.start)
        t = turn_by_index[idx]
        text = clean_text(" ".join(x.text for x in bucket))
        if not text: continue
        w_start = min(x.start for x in bucket)
        w_end = max(x.end for x in bucket)
        rows.append({
            "start": w_start, "end": w_end,
            "speaker_id": t.speaker,
            "role": label_by_speaker.get(t.speaker, t.speaker),
            "text": text,
        })
    rows.sort(key=lambda r: r["start"])
    return rows

def merge_same_speaker(rows: Iterable[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for r in rows:
        if not merged:
            merged.append(dict(r))
            continue
        prev = merged[-1]
        if prev["speaker_id"] == r["speaker_id"] and float(r["start"]) - float(prev["end"]) <= gap:
            prev["text"] = clean_text(prev["text"] + " " + r["text"])
            prev["end"] = max(float(prev["end"]), float(r["end"]))
        else:
            merged.append(dict(r))
    return merged

def format_lines(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in rows:
        a = fmt_mm_ss(float(r["start"]))
        b = fmt_mm_ss(float(r["end"]))
        lines.append(f"[{a} - {b}] {r['role']}:\n{r['text']}\n")
    return "\n".join(lines).strip() + "\n"

def align_combined_data(data: dict[str, Any], merge_gap: float = 0.9, agent_speaker: str | None = None, strict_two: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    turns = load_diar(data)
    words = load_words(data)
    
    if not turns: raise ValueError("No diarization turns found.")
    
    ranked_spk = unique_speakers_by_duration(turns)
    n_ids = len(ranked_spk)
    if n_ids != EXPECTED_PARTY_COUNT:
        msg = f"Diarization has {n_ids} distinct speaker ids; expected {EXPECTED_PARTY_COUNT}."
        if strict_two: raise ValueError(msg)
        warnings.append(msg)

    if not words: raise ValueError("No wordLevelTranscription found.")

    all_spk = {t.speaker for t in turns}
    if agent_speaker and agent_speaker in all_spk:
        label_by_speaker = {spk: "Customer" for spk in all_spk}
        label_by_speaker[agent_speaker] = "Agent"
    else:
        label_by_speaker = speaker_labels_two_party(turns)

    rows = build_rows(words, turns, label_by_speaker)
    rows = merge_same_speaker(rows, merge_gap)
    text_out = format_lines(rows)
    
    return {
        "text": text_out,
        "lines": rows,
        "merge_gap_seconds": merge_gap,
        "warnings": warnings,
    }