//
// Stem mode: how much of one part another part already plays.
//

#ifndef NoteMatching_h
#define NoteMatching_h

#include <algorithm>
#include <map>
#include <vector>

#include "NoteEvent.h"

namespace NoteMatching
{
/** Two notes are the same played note when they share a pitch and start this close together. */
inline constexpr double ONSET_TOLERANCE_SECONDS = 0.08;

/**
 * @return The fraction of inPart's notes that inOther also has (same pitch, onsets within
 *         ONSET_TOLERANCE_SECONDS), whatever instrument either names. 0 for an empty inPart.
 */
inline double coverage(const std::vector<NoteEvent>& inPart, const std::vector<NoteEvent>& inOther)
{
    if (inPart.empty() || inOther.empty()) {
        return 0.0;
    }

    // Onsets per pitch, sorted, so each lookup is a binary search.
    std::map<int, std::vector<double>> onsets;

    for (const NoteEvent& note: inOther) {
        onsets[note.pitch].push_back(note.startTime);
    }

    for (auto& entry: onsets) {
        std::sort(entry.second.begin(), entry.second.end());
    }

    std::size_t matched = 0;

    for (const NoteEvent& note: inPart) {
        const auto it = onsets.find(note.pitch);

        if (it == onsets.end()) {
            continue;
        }

        const std::vector<double>& times = it->second;
        const auto first = std::lower_bound(times.begin(), times.end(), note.startTime - ONSET_TOLERANCE_SECONDS);

        if (first != times.end() && *first <= note.startTime + ONSET_TOLERANCE_SECONDS) {
            matched++;
        }
    }

    return static_cast<double>(matched) / static_cast<double>(inPart.size());
}

/** @return inNotes split by instrument id. */
inline std::map<int, std::vector<NoteEvent>> byInstrument(const std::vector<NoteEvent>& inNotes)
{
    std::map<int, std::vector<NoteEvent>> parts;

    for (const NoteEvent& note: inNotes) {
        parts[note.program].push_back(note);
    }

    return parts;
}

/** What to do with a separated track's notes, given what the full-mix pass already shows. */
struct Merge {
    bool addTrack = false; // show the track as a row of its own
    std::vector<int> replacedInstruments; // full-mix rows it supersedes, to remove
};

/**
 * A full-mix row is the same part as the track when most of the row's notes are in the track.
 * The track then takes that row's place if it has at least as many notes -- the separated track
 * usually hears much more -- and is dropped otherwise, as a duplicate of a row already there.
 * A track that no row matches is something the full-mix pass missed, and is added.
 *
 * @param inTrack The track's notes, all under its own id.
 * @param inFullMix The full-mix pass's notes, by instrument.
 * @param inMinCoverage The share of a row's notes the track must also play to count as that row.
 */
inline Merge mergeTrack(const std::vector<NoteEvent>& inTrack,
                        const std::map<int, std::vector<NoteEvent>>& inFullMix,
                        double inMinCoverage = 0.5)
{
    Merge merge;

    if (inTrack.empty()) {
        return merge;
    }

    bool duplicate = false;

    for (const auto& [instrument, notes]: inFullMix) {
        if (coverage(notes, inTrack) < inMinCoverage) {
            continue;
        }

        if (inTrack.size() >= notes.size()) {
            merge.replacedInstruments.push_back(instrument);
        } else {
            duplicate = true;
        }
    }

    merge.addTrack = !duplicate || !merge.replacedInstruments.empty();

    if (!merge.addTrack) {
        merge.replacedInstruments.clear();
    }

    return merge;
}
} // namespace NoteMatching

#endif // NoteMatching_h
