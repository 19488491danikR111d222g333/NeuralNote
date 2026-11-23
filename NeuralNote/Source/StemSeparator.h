//
// Stem mode: splits the song into tracks with the Stems_to_FL Python tools before transcription.
//

#ifndef StemSeparator_h
#define StemSeparator_h

#include <atomic>
#include <functional>
#include <map>
#include <optional>

#include <JuceHeader.h>

/**
 * Runs the separation chain of Stems_to_FL (BS-RoFormer, Mel-RoFormer, De-Reverb, Demucs 6s ...)
 * as a child process, on the GPU through its own PyTorch environment, and reads back which tracks
 * it found.
 *
 * Where Stems_to_FL lives, in order of precedence:
 *   1. the NEURALNOTE_STEMS_DIR environment variable,
 *   2. the first line of stems_dir.txt in the NeuralNote folder (%APPDATA%\NeuralNote),
 *   3. DEFAULT_STEMS_DIR below.
 * Writing "off" as that first line turns stem mode off: NeuralNote transcribes the full mix as before.
 */
class StemSeparator
{
public:
    struct Tools {
        juce::File python; // env\Scripts\python.exe
        juce::File script; // stems_to_fl.py
    };

    struct Manifest {
        double bpm = 0.0;
        juce::String key;
        // Track name -> WAV at the song's own rate: lead_vocal, backing_vocals, drums, bass, guitar,
        // piano, woodwinds, other ... Only the tracks that were found and are not silent.
        std::map<juce::String, juce::File> stems;
        juce::File folder; // the <song>_FL folder, with the audio stems for the DAW as well
    };

    /** @return The Python environment and script, or nothing if stem mode is off or not installed. */
    static std::optional<Tools> findTools();

    /**
     * @return Where the stems of inSource go: <song>_FL next to it, like the Stems_to_FL batch file
     *         makes, or a folder under the NeuralNote directory when the song's folder cannot be
     *         written to.
     */
    static juce::File outputFolderFor(const juce::File& inSource);

    /**
     * Separates inSource into outFolder. Blocking -- minutes for a full song -- so off the message
     * thread. Reuses a previous separation of the same file. Kills the child process as soon as
     * inShouldCancel says so.
     *
     * @param inOnProgress Called with 0..1, from this thread.
     * @param outError Why it failed, short enough to show; empty on success or cancellation.
     */
    static std::optional<Manifest> run(const Tools& inTools,
                                       const juce::File& inSource,
                                       const juce::File& inOutFolder,
                                       const std::function<bool()>& inShouldCancel,
                                       const std::function<void(float)>& inOnProgress,
                                       juce::String& outError);

    /** @return The manifest Stems_to_FL wrote into inFolder, or nothing if it is missing or unreadable. */
    static std::optional<Manifest> readManifest(const juce::File& inFolder);
};

#endif // StemSeparator_h
