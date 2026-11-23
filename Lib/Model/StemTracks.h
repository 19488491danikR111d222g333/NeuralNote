//
// Stem mode: one sidebar row per separated track of the song.
//

#ifndef StemTracks_h
#define StemTracks_h

#include <cstdint>
#include <iterator>

#include "muscriptor/note.hpp"

/**
 * Instrument ids.
 *
 * Everywhere in the plugin an instrument is one integer: 0-127 for a melodic program decoded from
 * the full mix, msl::DRUM_PROGRAM (128) for drums, and -- in stem mode -- FIRST_STEM_ID and up for
 * a track separated out of the song before transcription (lead vocal, kick, bass...). The sidebar,
 * the piano roll colours, the synth and the MIDI file all key on that integer, so a stem only has
 * to be a new id to get its own row, colour, fader and MIDI track.
 *
 * A stem id is not a sound: soundProgramFor says which General MIDI program (or the drum kit) the
 * synth and the MIDI file use for it.
 */
enum class StemTrack : int {
    LeadVocal = 0,
    BackingVocals,
    Kick,
    Snare,
    Toms,
    HiHat,
    Ride,
    Crash,
    Percussion,
    Bass,
    Guitar,
    Keys,
    Winds,
    NumTracks
};

inline constexpr int FIRST_STEM_ID = msl::DRUM_PROGRAM + 1;
inline constexpr int NUM_STEM_TRACKS = static_cast<int>(StemTrack::NumTracks);
inline constexpr int NUM_INSTRUMENT_IDS = FIRST_STEM_ID + NUM_STEM_TRACKS;

struct StemTrackInfo {
    StemTrack track;
    const char* name;
    const char* abbreviation;
    std::uint32_t colour;
    int soundProgram; // General MIDI program, or msl::DRUM_PROGRAM
};

// In sidebar order. Colours follow the families of the full-mix instruments (voice cyan, drums
// slate, bass purple, guitars green, keys blue, winds chartreuse) with values none of them use.
inline constexpr StemTrackInfo STEM_TRACK_INFOS[] = {
    {StemTrack::LeadVocal, "Lead Vocal", "VOX", 0x5fd4f5, 53},
    {StemTrack::BackingVocals, "Backing Vocals", "BVX", 0x2f9fc4, 52},
    {StemTrack::Kick, "Kick", "KCK", 0xa9b4c8, msl::DRUM_PROGRAM},
    {StemTrack::Snare, "Snare", "SNR", 0x8d9ab3, msl::DRUM_PROGRAM},
    {StemTrack::Toms, "Toms", "TOM", 0x6e7d96, msl::DRUM_PROGRAM},
    {StemTrack::HiHat, "Hi-Hat", "HHT", 0xc4cbe0, msl::DRUM_PROGRAM},
    {StemTrack::Ride, "Ride", "RDE", 0x9aa6c4, msl::DRUM_PROGRAM},
    {StemTrack::Crash, "Crash", "CRS", 0xd6dbea, msl::DRUM_PROGRAM},
    {StemTrack::Percussion, "Percussion", "PRC", 0x58667d, msl::DRUM_PROGRAM},
    {StemTrack::Bass, "Bass", "BAS", 0xa84dff, 33},
    {StemTrack::Guitar, "Guitar", "GTR", 0x5fe0b8, 27},
    {StemTrack::Keys, "Keys", "KEY", 0x5a8cff, 0},
    {StemTrack::Winds, "Winds & Brass", "WND", 0xd4e37a, 73},
};

static_assert(std::size(STEM_TRACK_INFOS) == static_cast<std::size_t>(NUM_STEM_TRACKS),
              "One entry per StemTrack, in enumerator order");

constexpr bool stemTableInOrder()
{
    for (int i = 0; i < NUM_STEM_TRACKS; i++) {
        if (static_cast<int>(STEM_TRACK_INFOS[i].track) != i) {
            return false;
        }
    }

    return true;
}

static_assert(stemTableInOrder(), "STEM_TRACK_INFOS must follow the StemTrack enumerator order");

inline constexpr int stemId(StemTrack inTrack)
{
    return FIRST_STEM_ID + static_cast<int>(inTrack);
}

inline constexpr bool isStemId(int inId)
{
    return inId >= FIRST_STEM_ID && inId < NUM_INSTRUMENT_IDS;
}

inline constexpr const StemTrackInfo* stemInfoFor(int inId)
{
    return isStemId(inId) ? &STEM_TRACK_INFOS[inId - FIRST_STEM_ID] : nullptr;
}

/** @return The program the synth and the MIDI file play an instrument id with. */
inline constexpr int soundProgramFor(int inId)
{
    const StemTrackInfo* info = stemInfoFor(inId);
    return info != nullptr ? info->soundProgram : inId;
}

/** @return Whether the id is a drum instrument: full-mix drums or one of the drum stems. */
inline constexpr bool isDrumId(int inId)
{
    return soundProgramFor(inId) == msl::DRUM_PROGRAM;
}

/**
 * @return The position of an id in the sidebar: the stems first, in table order (vocals, drum
 *         pieces, bass, guitar, keys, winds), then whatever the model found in the rest of the mix,
 *         by program.
 */
inline constexpr int sidebarOrderFor(int inId)
{
    return isStemId(inId) ? inId - FIRST_STEM_ID : NUM_STEM_TRACKS + inId;
}

/**
 * Which drum stem a General MIDI drum key belongs to, for the drums stem: transcribed once as a
 * kit, then split by key so every piece gets its own row and MIDI track.
 */
inline constexpr StemTrack drumTrackForKey(int inKey)
{
    switch (inKey) {
        case 35:
        case 36:
            return StemTrack::Kick;
        case 37:
        case 38:
        case 39:
        case 40:
            return StemTrack::Snare;
        case 41:
        case 43:
        case 45:
        case 47:
        case 48:
        case 50:
            return StemTrack::Toms;
        case 42:
        case 44:
        case 46:
            return StemTrack::HiHat;
        case 51:
        case 53:
        case 59:
            return StemTrack::Ride;
        case 49:
        case 52:
        case 55:
        case 57:
            return StemTrack::Crash;
        default:
            return StemTrack::Percussion;
    }
}

#endif // StemTracks_h
