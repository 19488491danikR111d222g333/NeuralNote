//
// Created by Damien Ronssin on 02.06.2024.
//

#ifndef TranscriptionManager_h
#define TranscriptionManager_h

#include <optional>
#include <string>

#include <JuceHeader.h>
#include "MuscriptorEngine.h"
#include "StemSeparator.h"
#include "TranscriptionConstants.h"

class NeuralNoteAudioProcessor;
class NeuralNoteMainView;
class NeuralNoteEditor;

class TranscriptionManager : public Timer
{
public:
    explicit TranscriptionManager(NeuralNoteAudioProcessor* inProcessor);

    ~TranscriptionManager() override;

    void timerCallback() override;

    /** Transcribes the loaded audio from the start. AudioLoaded only. Message thread. */
    void startTranscription();

    /**
     * Continues a paused transcription where it stopped. Paused only. Tells the user instead when
     * the model it was started with is no longer installed. Message thread.
     */
    void resumeTranscription();

    /**
     * Stops the running transcription and keeps what it has transcribed, landing in Paused -- or in
     * AudioLoaded if there is nothing to resume from: a fresh run stopped before its first chunk
     * finished.
     */
    void pauseTranscription();

    /** Stops the running transcription and drops what it has transcribed, landing in AudioLoaded. */
    void discardTranscription();

    const std::vector<NoteEvent>& getNoteEventVector() const;

    /**
     * @return Phase and progress of the current/last transcription. While paused, the progress the
     *         run had reached at its resume point. Message thread.
     */
    MuscriptorEngine::Progress getTranscriptionProgress() const;

    /**
     * @return Time in seconds below which the note vector is complete, not merely correct. While a
     *         transcription runs or is paused it is the decode frontier: everything before it can be
     *         played and drawn, and nothing is known after it. Equal to the audio duration once finished.
     */
    double getFinalizedThrough() const;

    /** @return Whether the current transcription has been asked to stop. */
    bool isCancelRequested() const;

    /** @return Whether the running or last transcription splits the song into tracks first. */
    bool isStemJob() const { return mJobIsStem.load(); }

    void clear();

    /** @return The checkpoint the current transcription is from, running or finished. Message thread. */
    std::optional<ModelSize> getTranscriptionModelSize() const;

    /**
     * @return The transcription as an NnId::TranscriptionId tree once finished, as an
     *         NnId::PartialTranscriptionId tree while running or paused with at least one chunk done,
     *         or an invalid tree. Safe from any thread: reads only what was serialised ahead, under a lock.
     */
    ValueTree createStateTree() const;

    /**
     * Restores the transcription a saved full state holds, landing in PopulatedAudioAndMidiRegions
     * or Paused without going through Processing. No transcription, or an unreadable one, leaves
     * AudioLoaded. Expects the audio it was made from to be loaded already, and does nothing while a
     * job is running. Message thread.
     */
    void restoreFromStateTree(const ValueTree& inFullState);

private:
    /** Queues the job with mJob* set. Message thread. */
    void _launchJob();

    void _runModel();

    /**
     * Stem mode, on the job thread: separates the song with Stems_to_FL, then transcribes every
     * track on its own with the instruments that track can hold, each under its own instrument id
     * (StemTracks.h). Falls back to the full mix when the separation is not available or fails.
     */
    void _runStemJob();

    /** One track to transcribe, and how its notes become rows. */
    struct StemPlan {
        juce::String name;
        juce::File file;
        std::vector<msl::InstrumentGroup> instruments;
        std::function<bool(NoteEvent&)> mapper;
        bool dropSparseInstruments = false; // the leftover mix: instruments with a handful of notes are noise
    };

    static std::vector<StemPlan> _planStems(const StemSeparator::Manifest& inManifest);

    /** Stem mode's equivalent of _drainEngine. */
    void _drainStemJob();

    /** Stops every stem worker's engine. Any thread. */
    void _cancelStemEngines();

    /**
     * Applies mJobOutcome. Message thread only: everything downstream of it (the processor state,
     * the value tree, the UI) belongs to that thread.
     */
    void _handleFinishedJob();

    /** Takes whatever the engine has decoded since the last call, and passes it on. */
    void _drainEngine();

    /**
     * Re-derives the played and drawn notes from the raw ones and fans them out to the synth, the
     * mixer and the piano roll. Runs per decoded chunk, so a partial transcription is playable.
     */
    void _updatePostProcessing();

    /** Gives the synth a player for every instrument in the current notes. Message thread. */
    void _ensureSynthInstruments();

    /**
     * Catches up once the soundfont finishes loading, in case it was still loading the last time
     * _ensureSynthInstruments ran. It normally is not: a transcription takes minutes and the font
     * loads in about a second. Without this, a transcription that finishes inside that window would
     * leave every instrument silent for good, because nothing else re-scans the note list once the
     * font becomes ready.
     */
    void _catchUpSynthInstrumentsOnceFontReady();

    void _repaintPianoRoll();

    /**
     * Snapshots the transcription for createStateTree: finished, or partial with its resume point.
     * Message thread, once per decoded chunk and once when the run ends.
     */
    void _updateSavedTranscription();

    NeuralNoteAudioProcessor* mProcessor;

    MuscriptorEngine mMuscriptorEngine;

    // What the running job was launched with. Written by _launchJob and its callers before the job
    // is queued and read only by the job, so they need no synchronisation of their own.
    std::vector<msl::InstrumentGroup> mJobInstruments;
    ModelSize mJobModelSize = DEFAULT_MODEL_SIZE;
    ComputeDeviceChoice mJobDevice;
    std::string mJobResumeFrom;
    float mJobStartProgress = 0.0f;

    // The model's own output, accumulated chunk by chunk as the job decodes it. Message thread
    // only, and owned here rather than by the engine because it has to outlive it: the engine is
    // reset between runs, and this is what post-processing re-derives from and what the plugin
    // state persists.
    std::vector<NoteEvent> mRawNotes;

    // How many of mRawNotes a resumed job started with. Its final notes replace everything after.
    std::size_t mNotesBeforeJob = 0;

    // Where the library can continue from; empty unless the transcription is unfinished and at
    // least one chunk is in. Message thread only.
    std::string mResumePoint;

    // The run's progress at mResumePoint, in [0, 1].
    float mResumeProgress = 0.0f;

    // Set by discardTranscription, so the stopped job lands in AudioLoaded rather than Paused.
    bool mDiscardOnStop = false;

    // Message thread only. Set from the moment a job is launched, so it names a partial transcription too.
    std::optional<ModelSize> mTranscriptionModelSize;

    // What createStateTree writes, copied because hosts can ask for the state from any thread. An
    // empty resume point means a finished transcription.
    struct SavedTranscription {
        std::vector<NoteEvent> notes;
        ModelSize modelSize;
        double finalizedThrough;
        std::string resumePoint;
        float resumeProgress;
    };

    std::optional<SavedTranscription> mSavedTranscription;
    // mSavedTranscription's notes as JSON, built on the first createStateTree after they change.
    mutable std::optional<String> mSavedNotesJson;
    mutable CriticalSection mSavedTranscriptionLock;

    // How far mRawNotes is complete; see getFinalizedThrough.
    double mFinalizedThrough = 0.0;

    std::vector<NoteEvent> mPostProcessedNotes;

    // Written by the job, then published by mJobFinished.
    MuscriptorEngine::Outcome mJobOutcome = MuscriptorEngine::Outcome::Success;
    std::atomic<bool> mJobFinished = false;

    // True from the moment a job is queued until the message thread has applied its outcome. Only
    // there to make clear()'s "no job may be in flight" precondition assertable: it cannot be
    // derived from mThreadPool, which still reports the job as running when the outcome lands.
    std::atomic<bool> mJobActive = false;

    // Set once _catchUpSynthInstrumentsOnceFontReady has done its one catch-up pass, so it does not
    // rescan the note list on every timer tick for the rest of the session.
    bool mDidCatchUpSynthInstruments = false;

    // ---- stem mode. mJobIsStem is set on the message thread before the job is queued, and cleared
    // by the job itself if it falls back to the full mix.
    std::atomic<bool> mJobIsStem {false};
    std::atomic<bool> mStemCancel {false};
    std::optional<StemSeparator::Tools> mStemTools; // what the job runs; written before it is queued
    juce::File mStemSource;
    juce::File mStemOutFolder;
    std::atomic<int> mStemPhase {0}; // 1 once the extra tracks are being transcribed
    std::atomic<bool> mStemFullMixDone {false};
    std::atomic<float> mStemSeparationProgress {-1.0f};
    std::atomic<int> mStemIndex {0};
    std::atomic<int> mStemCount {0};
    std::atomic<double> mStemBpm {0.0};
    juce::String mStemError; // written by the job before mJobFinished

    // Tracks already transcribed, published by the job after each one. mStemGeneration counts them.
    juce::CriticalSection mStemLock;
    std::vector<NoteEvent> mStemCompleted;
    int mStemGeneration = 0;
    int mStemDoneCount = 0;
    // One engine per worker while tracks are being transcribed, read for progress and cancellation.
    std::vector<std::unique_ptr<MuscriptorEngine>> mStemEngines;

    // Message thread: the completed tracks as last seen, and the one being decoded.
    int mStemSeenGeneration = -1;
    std::vector<NoteEvent> mStemCompletedCopy;
    std::vector<NoteEvent> mStemStreaming;
    double mStemHorizon = 0.0; // how far the track being decoded has got

    ThreadPool mThreadPool;
    std::function<void()> mJobLambda;
    std::function<void()> mStemJobLambda;
};

#endif //TranscriptionManager_h
