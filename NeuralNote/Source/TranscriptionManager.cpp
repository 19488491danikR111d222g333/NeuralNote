//
// Created by Damien Ronssin on 02.06.2024.
//

#include "TranscriptionManager.h"
#include "PluginProcessor.h"
#include "NeuralNoteMainView.h"
#include "InstrumentSelection.h"
#include "NNFileUtils.h"
#include "NnGlobalSettings.h"
#include "NnLogger.h"
#include "TranscriptionConstants.h"
#include "AudioUtils.h"
#include "NnId.h"
#include "StemTracks.h"
#include "NoteMatching.h"

#include <map>
#include <thread>

namespace
{
// The layout of the saved TRANSCRIPTION tree and its notes.
constexpr int TRANSCRIPTION_FORMAT_VERSION = 1;

// The layout of the saved PARTIAL_TRANSCRIPTION tree.
constexpr int PARTIAL_TRANSCRIPTION_FORMAT_VERSION = 1;

// Note times are saved to the microsecond. JUCE trims the trailing zeros, so a time on the model's
// 10 ms grid is written as e.g. "0.33".
constexpr int SAVED_TIME_DECIMAL_PLACES = 6;

/**
 * Reads the notes notesToJson wrote. Any malformed entry rejects the whole list; a note starting
 * outside [0, inDuration] is dropped. The end is not bounded: a note left open at the end of the
 * audio ends 10 ms after its onset, which can be past the last sample.
 */
std::optional<std::vector<NoteEvent>> parseNotes(const String& inJson, double inDuration)
{
    const var parsed = JSON::parse(inJson);
    const Array<var>* entries = parsed.getArray();

    if (entries == nullptr) {
        return std::nullopt;
    }

    std::vector<NoteEvent> notes;
    notes.reserve(static_cast<std::size_t>(entries->size()));

    for (const var& entry: *entries) {
        const Array<var>* fields = entry.getArray();

        if (fields == nullptr || fields->size() != 4) {
            return std::nullopt;
        }

        NoteEvent note;
        note.startTime = fields->getReference(0);
        note.endTime = fields->getReference(1);
        note.pitch = fields->getReference(2);
        note.program = fields->getReference(3);
        note.amplitude = FIXED_NOTE_AMPLITUDE;

        const bool valid_times =
            std::isfinite(note.startTime) && std::isfinite(note.endTime) && note.endTime >= note.startTime;
        const bool valid_pitch = note.pitch >= MIN_MIDI_NOTE && note.pitch <= MAX_MIDI_NOTE;
        const bool valid_program = note.program >= 0 && note.program < NUM_INSTRUMENT_IDS;

        if (!valid_times || !valid_pitch || !valid_program) {
            return std::nullopt;
        }

        if (note.startTime >= 0.0 && note.startTime <= inDuration) {
            notes.push_back(note);
        }
    }

    return notes;
}

String notesToJson(const std::vector<NoteEvent>& inNotes)
{
    Array<var> entries;
    entries.ensureStorageAllocated(static_cast<int>(inNotes.size()));

    for (const NoteEvent& note: inNotes) {
        entries.add(Array<var> {note.startTime, note.endTime, note.pitch, note.program});
    }

    const auto format =
        JSON::FormatOptions {}.withSpacing(JSON::Spacing::none).withMaxDecimalPlaces(SAVED_TIME_DECIMAL_PLACES);

    return JSON::toString(entries, format);
}

const char* outcomeToString(MuscriptorEngine::Outcome inOutcome)
{
    switch (inOutcome) {
        case MuscriptorEngine::Outcome::Success:
            return "success";
        case MuscriptorEngine::Outcome::Cancelled:
            return "cancelled";
        case MuscriptorEngine::Outcome::Failed:
            return "failed";
        case MuscriptorEngine::Outcome::CannotResume:
            return "cannot resume";
    }

    return "unknown";
}
} // namespace

TranscriptionManager::TranscriptionManager(NeuralNoteAudioProcessor* inProcessor)
    : mProcessor(inProcessor)
    , mThreadPool(1)
{
    mJobLambda = [this] { _runModel(); };
    mStemJobLambda = [this] { _runStemJob(); };

    startTimerHz(30);
}

TranscriptionManager::~TranscriptionManager()
{
    stopTimer();

    // ~ThreadPool waits 5 s and then kills the thread by force. Cancel and wait here instead, long
    // enough for GPU backend initialisation, which cannot be interrupted and took about 20 s on
    // the first run after a new build.
    mStemCancel = true;
    _cancelStemEngines();
    mMuscriptorEngine.cancel();

    const bool jobs_finished = mThreadPool.removeAllJobs(true, 30000);
    jassertquiet(jobs_finished);
}

void TranscriptionManager::timerCallback()
{
    _catchUpSynthInstrumentsOnceFontReady();

    if (mJobFinished.exchange(false)) {
        _handleFinishedJob();
        return;
    }

    if (mJobActive) {
        if (mJobIsStem) {
            _drainStemJob();
        } else {
            _drainEngine();
        }
    }
}

void TranscriptionManager::_drainEngine()
{
    // Gated on the drain having something, so this runs at the model's pace -- once per 5 s of
    // audio -- rather than at the timer's.
    if (mMuscriptorEngine.drainNewNotes(mRawNotes, mFinalizedThrough, mResumePoint, mResumeProgress)) {
        _updateSavedTranscription();
        _updatePostProcessing();
        _repaintPianoRoll();
    }
}

void TranscriptionManager::_runModel()
{
    mJobOutcome = mMuscriptorEngine.transcribeToMIDI(
        mJobModelSize,
        mJobDevice,
        mProcessor->getSourceAudioManager()->getDownsampledSourceAudioForTranscription().getWritePointer(0),
        mProcessor->getSourceAudioManager()->getNumSamplesDownAcquired(),
        mJobInstruments,
        mJobResumeFrom,
        mJobStartProgress);

    // Post-processing and the synth handoff belong to the message thread, which has been doing
    // both per chunk while this ran. All that is left here is to say so.
    //
    // Published last: the message thread treats this as the signal that the job is done with
    // everything it owns, and clearing while a job is in flight would race with it.
    mJobFinished = true;
}

void TranscriptionManager::_handleFinishedJob()
{
    using Outcome = MuscriptorEngine::Outcome;

    jassert(MessageManager::getInstance()->isThisTheMessageThread());

    // The job published its outcome as its last act, so nothing on the pool thread can touch the
    // state this manager owns from here on. Dropped before clear() runs below, which asserts it.
    mJobActive = false;

    if (mJobIsStem) {
        const bool keep = mJobOutcome == Outcome::Success && !mDiscardOnStop;

        NnLogger::write(NnLogger::Level::Info,
                        String("TranscriptionManager: stem job finished: ") + outcomeToString(mJobOutcome)
                            + (keep ? ", kept" : ", cleared"));

        if (keep) {
            {
                const ScopedLock sl(mStemLock);
                mRawNotes = mStemCompleted;
            }

            mStemStreaming.clear();
            mStemCompletedCopy.clear();
            mStemSeenGeneration = -1;
            mFinalizedThrough = mProcessor->getSourceAudioManager()->getAudioSampleDuration();
            mResumePoint.clear();
            mResumeProgress = 0.0f;
            _updateSavedTranscription();
            _updatePostProcessing();
            mProcessor->setStateToPopulatedAudioAndMidiRegions();
            _repaintPianoRoll();

            // The song's own tempo, so the exported MIDI sits on the DAW's grid.
            if (const double bpm = mStemBpm.load(); bpm > 0.0) {
                mProcessor->getValueTree().setProperty(NnId::ExportTempoId, bpm, nullptr);
            }

            if (mStemError.isNotEmpty()) {
                NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::NoIcon,
                                                      "Some tracks were not transcribed.",
                                                      mStemError);
            }

            return;
        }

        mProcessor->clearTranscription();

        if (mJobOutcome == Outcome::Failed) {
            NativeMessageBox::showMessageBoxAsync(
                MessageBoxIconType::NoIcon,
                "Transcription failed.",
                mStemError.isEmpty() ? String("The transcription model could not be loaded or run.") : mStemError);
        }

        return;
    }

    // Stem mode was on but the separation failed, and the job transcribed the full mix instead.
    if (mStemError.isNotEmpty()) {
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::NoIcon,
                                              "Track separation failed.",
                                              mStemError + "\n\nThe full mix was transcribed instead.");
        mStemError.clear();
    }

    if (mJobOutcome == Outcome::Success) {
        // Replaces what this job streamed rather than extending it: transcribe()'s own result is
        // authoritative, and the streamed one is missing any note the model never closed.
        auto final_notes = mMuscriptorEngine.takeFinalNotes();
        mRawNotes.resize(mNotesBeforeJob);
        mRawNotes.insert(mRawNotes.end(), final_notes.begin(), final_notes.end());
        mFinalizedThrough = mProcessor->getSourceAudioManager()->getAudioSampleDuration();
        mResumePoint.clear();
        mResumeProgress = 0.0f;
        _updateSavedTranscription();
        _updatePostProcessing();
        mProcessor->setStateToPopulatedAudioAndMidiRegions();
        _repaintPianoRoll();
        NnLogger::write(NnLogger::Level::Info, "TranscriptionManager: job finished: success");
        return;
    }

    // Read before clearTranscription(), which resets the engine and with it the message.
    const String reason = mMuscriptorEngine.getLastErrorMessage();
    bool keep = !mDiscardOnStop && mJobOutcome != Outcome::CannotResume;

    if (keep) {
        // Whatever landed between the last timer tick and the stop belongs to the paused transcription.
        _drainEngine();
        keep = !mResumePoint.empty();
    }

    NnLogger::write(NnLogger::Level::Info,
                    String("TranscriptionManager: job finished: ") + outcomeToString(mJobOutcome)
                        + (keep ? ", can be resumed" : ", cleared"));

    if (keep) {
        mProcessor->setStateToPaused();
        _repaintPianoRoll();
    } else {
        // Back to where the transcribe button was, with the audio still loaded: stopping a run
        // the user misconfigured should not cost them the file as well.
        mProcessor->clearTranscription();
    }

    if (mJobOutcome == Outcome::Failed) {
        String message = reason.isEmpty() ? String("The transcription model could not be loaded or run.")
                                          : "The transcription model could not be loaded or run: " + reason + ".";

        if (keep) {
            message += " What was transcribed so far is kept, and can be resumed.";
        }

        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::NoIcon, "Transcription failed.", message);
    }

    if (mJobOutcome == Outcome::CannotResume) {
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::NoIcon,
                                              "Transcription could not be resumed.",
                                              "The paused transcription does not match the loaded audio. "
                                              "Transcribe again to start over.");
    }
}

void TranscriptionManager::_updatePostProcessing()
{
    jassert(mProcessor->hasTranscription());

    if (mProcessor->hasTranscription()) {
        mPostProcessedNotes = mRawNotes;

        NoteEvent::mergeOverlappingNotesWithSamePitch(mPostProcessedNotes);

        // Before the notes reach the scheduler, so no note can arrive at the synth for an
        // instrument that has no player yet. Creating one allocates and touches the list the audio
        // thread walks, so it happens here on the message thread rather than on demand.
        _ensureSynthInstruments();

        // After the synth has players for them, so the faders it pushes land somewhere.
        mProcessor->getInstrumentMixer()->rebuildFromNotes(mPostProcessedNotes);

        // For the synth. A copy, because mPostProcessedNotes is what the piano roll draws and the
        // scheduler takes ownership of what it is given.
        auto notes_to_play = mPostProcessedNotes;
        mProcessor->getPlayer()->getSynthController()->setNotes(notes_to_play);
    }
}

void TranscriptionManager::_ensureSynthInstruments()
{
    jassert(MessageManager::getInstance()->isThisTheMessageThread());

    auto* synth = mProcessor->getPlayer()->getInstrumentSynth();

    // A transcription streams in, so the set of instruments grows as it decodes; this runs once per
    // chunk and adds whatever is new. Scanning the whole note list each time is a pass over
    // something the piano roll redraws entirely anyway, and it keeps the "which instruments exist"
    // question answerable from the notes alone rather than from a second, driftable record.
    std::array<bool, NUM_INSTRUMENT_IDS> seen {};

    for (const NoteEvent& note: mPostProcessedNotes) {
        if (note.program >= 0 && note.program < NUM_INSTRUMENT_IDS) {
            seen[static_cast<std::size_t>(note.program)] = true;
        }
    }

    // Under the callback lock, for the same reason setNotes takes it: this appends to the list the
    // audio thread walks every block.
    const ScopedLock sl(mProcessor->getCallbackLock());

    for (std::size_t program = 0; program < seen.size(); program++) {
        if (seen[program]) {
            synth->ensureInstrument(static_cast<int>(program));
        }
    }
}

void TranscriptionManager::_catchUpSynthInstrumentsOnceFontReady()
{
    if (mDidCatchUpSynthInstruments || !mProcessor->hasTranscription()) {
        return;
    }

    if (mProcessor->getPlayer()->getInstrumentSynth()->isReady()) {
        // Whatever _ensureSynthInstruments has already run so far may have found the synth not
        // ready yet and done nothing; this repeats it exactly once now that it is, for every
        // instrument the current notes name. A no-op for any instrument that already exists.
        _ensureSynthInstruments();
        mDidCatchUpSynthInstruments = true;
    }
}

const std::vector<NoteEvent>& TranscriptionManager::getNoteEventVector() const
{
    return mPostProcessedNotes;
}

MuscriptorEngine::Progress TranscriptionManager::getTranscriptionProgress() const
{
    if (mJobActive.load() && mJobIsStem.load()) {
        // One bar for the whole job: the full-mix pass, the split running beside it, then the
        // extra tracks.
        float full_mix = 1.0f;

        if (!mStemFullMixDone.load()) {
            const MuscriptorEngine::Progress progress = mMuscriptorEngine.getProgress();
            full_mix = progress.phase == MuscriptorEngine::Phase::Transcribing ? std::max(0.0f, progress.fraction) : 0.0f;
        }

        const float separation = std::max(0.0f, mStemSeparationProgress.load());

        const ScopedLock sl(mStemLock);
        float tracks = 0.0f;

        if (mStemPhase.load() == 1) {
            const int count = std::max(1, mStemCount.load());
            float sum = static_cast<float>(mStemDoneCount);

            for (const auto& engine: mStemEngines) {
                const MuscriptorEngine::Progress progress = engine->getProgress();

                if (progress.phase == MuscriptorEngine::Phase::Transcribing && progress.fraction < 1.0f) {
                    sum += std::max(0.0f, progress.fraction);
                }
            }

            tracks = sum / static_cast<float>(count);
        }

        const float overall = 0.4f * full_mix + 0.2f * separation + 0.4f * tracks;
        return {MuscriptorEngine::Phase::Transcribing, std::clamp(overall, 0.0f, 1.0f)};
    }

    if (mProcessor->getState() == Paused) {
        return {MuscriptorEngine::Phase::Transcribing, mResumeProgress};
    }

    return mMuscriptorEngine.getProgress();
}

double TranscriptionManager::getFinalizedThrough() const
{
    return mFinalizedThrough;
}

void TranscriptionManager::pauseTranscription()
{
    NnLogger::write(NnLogger::Level::Info, "TranscriptionManager: pause requested");
    mStemCancel = true;
    _cancelStemEngines();
    mMuscriptorEngine.cancel();
}

void TranscriptionManager::discardTranscription()
{
    NnLogger::write(NnLogger::Level::Info, "TranscriptionManager: discard requested");
    mDiscardOnStop = true;
    mStemCancel = true;
    _cancelStemEngines();
    mMuscriptorEngine.cancel();
}

bool TranscriptionManager::isCancelRequested() const
{
    return mMuscriptorEngine.isCancelRequested() || (mJobActive.load() && mJobIsStem.load() && mStemCancel.load());
}

void TranscriptionManager::clear()
{
    // Resets state a running job owns, so it must not be called while one is in flight. Every
    // path here either runs before the job is launched or after _handleFinishedJob.
    jassert(!mJobActive.load());

    mMuscriptorEngine.reset();
    mJobFinished = false;
    mRawNotes.clear();
    mRawNotes.shrink_to_fit();
    mNotesBeforeJob = 0;
    mResumePoint.clear();
    mResumeProgress = 0.0f;
    mDiscardOnStop = false;
    mFinalizedThrough = 0.0;
    mPostProcessedNotes.clear();
    mTranscriptionModelSize.reset();
    mJobIsStem = false;
    mStemCompletedCopy.clear();
    mStemStreaming.clear();
    mStemSeenGeneration = -1;

    {
        const ScopedLock sl(mSavedTranscriptionLock);
        mSavedTranscription.reset();
        mSavedNotesJson.reset();
    }

    mProcessor->getInstrumentMixer()->clear();

    // Otherwise the synth keeps playing the notes of the transcription that was just thrown away.
    std::vector<NoteEvent> no_notes;
    mProcessor->getPlayer()->getSynthController()->setNotes(no_notes);
    mProcessor->getPlayer()->getSynthController()->stopAllNotes();

    {
        // Drops the instruments and their faders with the transcription that named them. Under the
        // callback lock: it destroys what the audio thread renders from.
        const ScopedLock sl(mProcessor->getCallbackLock());
        mProcessor->getPlayer()->getInstrumentSynth()->reset();
    }
}

void TranscriptionManager::startTranscription()
{
    jassert(MessageManager::getInstance()->isThisTheMessageThread());

    // The Transcribe button only hides on the next timer tick, so a second click can arrive while
    // the first job is already running. Everything below writes state that job owns.
    if (mProcessor->getState() != AudioLoaded || mJobActive.load()) {
        return;
    }

    // The stored size is a preference: when that checkpoint is missing and another is there, the
    // run uses the one that is there.
    const std::optional<ModelSize> model_size = NNFileUtils::getInstalledModelSize(NnGlobalSettings::getModelSize());

    // The checkpoint went since the button was shown. The model panel's next poll replaces the button.
    if (!model_size.has_value()) {
        return;
    }

    // Have at least one second to transcribe
    if (mProcessor->getSourceAudioManager()->getNumSamplesDownAcquired() < 1 * TRANSCRIPTION_SAMPLE_RATE) {
        mProcessor->clear();
        return;
    }

    // Nothing is transcribed yet, and the piano roll is about to start drawing what arrives.
    mRawNotes.clear();
    mNotesBeforeJob = 0;
    mResumePoint.clear();
    mResumeProgress = 0.0f;
    mFinalizedThrough = 0.0;
    mPostProcessedNotes.clear();

    // Every instrument this run finds is a new one, and should start at unity and unmuted rather
    // than inherit a fader from whatever was transcribed before.
    mProcessor->getInstrumentMixer()->resetStoredSettings();

    mJobModelSize = *model_size;
    mTranscriptionModelSize = *model_size;
    mJobResumeFrom.clear();
    mJobStartProgress = 0.0f;

    // Stem mode: split the song into tracks first, when Stems_to_FL is installed and the audio is a
    // file (a dropped song, or a finished recording).
    mJobIsStem = false;
    mStemTools.reset();
    const File source = mProcessor->getSourceAudioManager()->getSourceFile();

    if (source.existsAsFile()) {
        mStemTools = StemSeparator::findTools();

        if (mStemTools.has_value()) {
            mJobIsStem = true;
            mStemSource = source;
            mStemOutFolder = StemSeparator::outputFolderFor(source);
        }
    }

    _launchJob();
}

void TranscriptionManager::resumeTranscription()
{
    jassert(MessageManager::getInstance()->isThisTheMessageThread());

    if (mProcessor->getState() != Paused || mJobActive.load() || mResumePoint.empty()
        || !mTranscriptionModelSize.has_value()) {
        return;
    }

    // Another checkpoint would continue another model's transcription.
    if (!NNFileUtils::isModelInstalled(*mTranscriptionModelSize)) {
        const String message = "This transcription was started with the "
                               + String(modelSizeToDisplayName(*mTranscriptionModelSize))
                               + " model. Download it from the model menu to resume.";
        NnLogger::write(NnLogger::Level::Warning, "TranscriptionManager: Model not installed. " + message);
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::NoIcon, "Model not installed.", message);
        return;
    }

    // The engine's final notes replace everything this job streams, and nothing before.
    mNotesBeforeJob = mRawNotes.size();

    mJobModelSize = *mTranscriptionModelSize;
    mJobResumeFrom = mResumePoint;
    mJobStartProgress = mResumeProgress;

    _launchJob();
}

void TranscriptionManager::_launchJob()
{
    // Armed before the Processing state is published, not after: that state is what makes the
    // pause button live, and reset() is what clears a pending cancellation request. Doing it in
    // this order means no click can be discarded, without having to argue about which thread
    // observes what.
    mMuscriptorEngine.reset();
    mDiscardOnStop = false;

    // Read here, not in the job: the selection lives on the state tree, which belongs to this
    // thread. The job only ever sees the copy. It cannot change while a transcription exists, so a
    // resumed run gets the selection its resume point was made with.
    mJobInstruments = InstrumentSelection::get(mProcessor->getValueTree());
    mJobDevice = NnGlobalSettings::getComputeDevice();

    NnLogger::write(
        NnLogger::Level::Info,
        String("TranscriptionManager: ") + (mJobResumeFrom.empty() ? "starting" : "resuming") + ": model "
            + modelSizeToString(mJobModelSize) + ", device "
            + (mJobDevice.isAuto() ? String("auto") : String(mJobDevice.name) + " #" + String(mJobDevice.ordinal))
            + ", " + (mJobInstruments.empty() ? String("all") : String(mJobInstruments.size())) + " instruments, "
            + String(mProcessor->getSourceAudioManager()->getAudioSampleDuration(), 1) + " s of audio");

    if (mJobIsStem) {
        mStemFullMixDone = false;
        mStemCancel = false;
        mStemPhase = 0;
        mStemSeparationProgress = -1.0f;
        mStemIndex = 0;
        mStemCount = 0;
        mStemBpm = 0.0;
        mStemSeenGeneration = -1;
        mStemCompletedCopy.clear();
        mStemStreaming.clear();
        mStemHorizon = 0.0;

        const ScopedLock sl(mStemLock);
        mStemCompleted.clear();
        mStemGeneration = 0;
    }

    mStemError.clear();

    mProcessor->setStateToProcessing();

    mJobActive = true;
    mThreadPool.addJob(mJobIsStem ? mStemJobLambda : mJobLambda);
}

std::optional<ModelSize> TranscriptionManager::getTranscriptionModelSize() const
{
    return mTranscriptionModelSize;
}

ValueTree TranscriptionManager::createStateTree() const
{
    const ScopedLock sl(mSavedTranscriptionLock);

    if (!mSavedTranscription.has_value()) {
        return {};
    }

    const SavedTranscription& saved = *mSavedTranscription;

    if (!mSavedNotesJson.has_value()) {
        mSavedNotesJson = notesToJson(saved.notes);
    }

    const bool finished = saved.resumePoint.empty();

    // A separate node for a partial one, so an older version that knows only TRANSCRIPTION skips it
    // rather than showing it as finished.
    ValueTree tree(finished ? NnId::TranscriptionId : NnId::PartialTranscriptionId);
    tree.setProperty(NnId::TranscriptionFormatVersionId,
                     finished ? TRANSCRIPTION_FORMAT_VERSION : PARTIAL_TRANSCRIPTION_FORMAT_VERSION,
                     nullptr);
    tree.setProperty(NnId::TranscriptionModelSizeId, modelSizeToString(saved.modelSize), nullptr);
    tree.setProperty(NnId::TranscriptionNotesId, *mSavedNotesJson, nullptr);

    if (!finished) {
        tree.setProperty(NnId::FinalizedThroughId, saved.finalizedThrough, nullptr);
        tree.setProperty(NnId::ResumePointId, String(saved.resumePoint), nullptr);
        tree.setProperty(NnId::ResumeProgressId, saved.resumeProgress, nullptr);
    }

    return tree;
}

void TranscriptionManager::restoreFromStateTree(const ValueTree& inFullState)
{
    if (mJobActive.load()) {
        return;
    }

    // Replaced even when the restored state has no transcription: the audio path may not have
    // changed, in which case nothing else would drop the notes on screen.
    if (mProcessor->getState() == PopulatedAudioAndMidiRegions || mProcessor->getState() == Paused) {
        mProcessor->clearTranscription();
    }

    if (mProcessor->getState() != AudioLoaded) {
        return;
    }

    ValueTree tree = inFullState.getChildWithName(NnId::TranscriptionId);
    int format_version = TRANSCRIPTION_FORMAT_VERSION;

    if (!tree.isValid()) {
        tree = inFullState.getChildWithName(NnId::PartialTranscriptionId);
        format_version = PARTIAL_TRANSCRIPTION_FORMAT_VERSION;
    }

    const bool finished = tree.hasType(NnId::TranscriptionId);
    const std::string resume_point = tree.getProperty(NnId::ResumePointId).toString().toStdString();

    if (!tree.isValid() || static_cast<int>(tree.getProperty(NnId::TranscriptionFormatVersionId)) != format_version
        || (!finished && resume_point.empty())) {
        return;
    }

    const double duration = mProcessor->getSourceAudioManager()->getAudioSampleDuration();
    auto notes = parseNotes(tree.getProperty(NnId::TranscriptionNotesId).toString(), duration);

    const std::optional<ModelSize> model_size =
        modelSizeFromString(tree.getProperty(NnId::TranscriptionModelSizeId).toString().toStdString());

    // Resuming with another checkpoint would mix two models' notes. A finished one only shows the name.
    if (!notes.has_value() || (!finished && !model_size.has_value())) {
        return;
    }

    mRawNotes = std::move(*notes);
    mTranscriptionModelSize = model_size.value_or(DEFAULT_MODEL_SIZE);

    if (finished) {
        mFinalizedThrough = duration;
    } else {
        mFinalizedThrough = std::clamp(static_cast<double>(tree.getProperty(NnId::FinalizedThroughId)), 0.0, duration);
        mResumePoint = resume_point;
        mResumeProgress = std::clamp(static_cast<float>(tree.getProperty(NnId::ResumeProgressId)), 0.0f, 1.0f);
    }

    _updateSavedTranscription();

    // Before post-processing, which only runs on a transcription the processor says it has.
    if (finished) {
        mProcessor->setStateToPopulatedAudioAndMidiRegions();
    } else {
        mProcessor->setStateToPaused();
    }

    _updatePostProcessing();
    _repaintPianoRoll();
}

void TranscriptionManager::_updateSavedTranscription()
{
    jassert(MessageManager::getInstance()->isThisTheMessageThread());
    jassert(mTranscriptionModelSize.has_value());

    SavedTranscription saved {mRawNotes, *mTranscriptionModelSize, mFinalizedThrough, mResumePoint, mResumeProgress};

    const ScopedLock sl(mSavedTranscriptionLock);
    mSavedTranscription = std::move(saved);
    mSavedNotesJson.reset();
}

void TranscriptionManager::_repaintPianoRoll()
{
    auto* main_view = mProcessor->getNeuralNoteMainView();

    if (main_view) {
        main_view->repaintPianoRoll();
    }
}

std::vector<TranscriptionManager::StemPlan> TranscriptionManager::_planStems(const StemSeparator::Manifest& inManifest)
{
    using G = msl::InstrumentGroup;

    const auto as_track = [](StemTrack inTrack) {
        const int id = stemId(inTrack);
        return std::function<bool(NoteEvent&)>([id](NoteEvent& ioNote) {
            ioNote.program = id;
            return true;
        });
    };

    // The drums stem is transcribed once as a kit, and every hit goes to the row of its piece.
    const auto drum_pieces = std::function<bool(NoteEvent&)>([](NoteEvent& ioNote) {
        if (!ioNote.isDrum()) {
            return false;
        }

        ioNote.program = stemId(drumTrackForKey(ioNote.pitch));
        return true;
    });

    // other.wav: whatever the model hears there is one row. Drum hits leaking into it are already in
    // the drum rows, and would show as nonsense pitches in a melodic one.
    const auto other_track = std::function<bool(NoteEvent&)>([](NoteEvent& ioNote) {
        if (ioNote.isDrum()) {
            return false;
        }

        ioNote.program = stemId(StemTrack::Other);
        return true;
    });

    const std::vector<G> voice = {G::Voice};
    const std::vector<G> bass = {G::ElectricBass, G::AcousticBass, G::Contrabass, G::SynthLead};
    const std::vector<G> guitar = {G::CleanElectricGuitar, G::DistortedElectricGuitar, G::AcousticGuitar};
    const std::vector<G> keys = {G::AcousticPiano, G::ElectricPiano, G::Organ, G::ChromaticPercussion};
    const std::vector<G> winds = {G::Flutes,
                                  G::Oboe,
                                  G::EnglishHorn,
                                  G::Clarinet,
                                  G::Bassoon,
                                  G::SopranoAndAltoSax,
                                  G::TenorSax,
                                  G::BaritoneSax,
                                  G::Trumpet,
                                  G::Trombone,
                                  G::FrenchHorn,
                                  G::BrassSection,
                                  G::Tuba};

    // Track name in the manifest -> what to look for in it and which row it fills. Reverb, crowd
    // and the full instrumental have no notes worth having, so they are not here.
    struct Rule {
        const char* name;
        std::vector<G> instruments;
        std::function<bool(NoteEvent&)> mapper;
        bool dropSparse;
    };

    const std::vector<Rule> rules = {
        {"lead_vocal", voice, as_track(StemTrack::LeadVocal), false},
        {"vocals", voice, as_track(StemTrack::LeadVocal), false}, // when lead/backing could not be split
        {"backing_vocals", voice, as_track(StemTrack::BackingVocals), false},
        {"backing_female", voice, as_track(StemTrack::BackingVocals), false},
        {"backing_male", voice, as_track(StemTrack::BackingVocals), false},
        {"drums", {G::Drums}, drum_pieces, false},
        {"bass", bass, as_track(StemTrack::Bass), false},
        {"guitar", guitar, as_track(StemTrack::Guitar), false},
        {"piano", keys, as_track(StemTrack::Keys), false},
        {"woodwinds", winds, as_track(StemTrack::Winds), false},
        // other.wav, under the name Stems_to_FL gives it for this version: one row, the model picks what
        // it hears. (Older builds read "other" and made a row per instrument they imagined there.)
        {"other_one", {}, other_track, false},
    };

    std::vector<StemPlan> plans;

    for (const Rule& rule: rules) {
        const auto it = inManifest.stems.find(rule.name);

        if (it == inManifest.stems.end()) {
            continue;
        }

        // "vocals" only stands in for a lead vocal that is missing.
        if (String(rule.name) == "vocals" && inManifest.stems.count("lead_vocal") > 0) {
            continue;
        }

        plans.push_back({rule.name, it->second, rule.instruments, rule.mapper, rule.dropSparse});
    }

    return plans;
}

void TranscriptionManager::_runStemJob()
{
    using Outcome = MuscriptorEngine::Outcome;

    // A row with fewer notes than this is something the model barely heard -- the one-note
    // "Synth Lead" -- and is left out rather than shown as an empty-looking row.
    constexpr int MIN_NOTES_PER_INSTRUMENT = 20;

    const auto drop_sparse_rows = [](std::vector<NoteEvent>& ioNotes) {
        std::map<int, int> counts;

        for (const NoteEvent& note: ioNotes) {
            counts[note.program]++;
        }

        ioNotes.erase(std::remove_if(ioNotes.begin(),
                                     ioNotes.end(),
                                     [&counts](const NoteEvent& inNote) {
                                         return counts[inNote.program] < MIN_NOTES_PER_INSTRUMENT;
                                     }),
                      ioNotes.end());
    };

    jassert(mStemTools.has_value());

    // 1. The split into tracks runs in the background, on its own thread, the whole time ...
    String separation_error;
    std::optional<StemSeparator::Manifest> manifest;

    std::thread separation([&] {
        manifest = StemSeparator::run(
            *mStemTools,
            mStemSource,
            mStemOutFolder,
            [this] { return mStemCancel.load(); },
            [this](float inFraction) { mStemSeparationProgress = inFraction; },
            separation_error);
    });

    // The split tracks are only a working copy: nothing is left behind afterwards.
    const ScopeGuard remove_working_copy {[this] {
        mStemOutFolder.deleteRecursively();
    }};

    // 2. ... while the ordinary NeuralNote pass transcribes the whole mix, exactly as before, and
    // streams into the piano roll from the start of the song.
    const Outcome full_mix_outcome = mMuscriptorEngine.transcribeToMIDI(
        mJobModelSize,
        mJobDevice,
        mProcessor->getSourceAudioManager()->getDownsampledSourceAudioForTranscription().getWritePointer(0),
        mProcessor->getSourceAudioManager()->getNumSamplesDownAcquired(),
        mJobInstruments,
        {},
        0.0f);

    std::vector<NoteEvent> full_mix;

    if (full_mix_outcome == Outcome::Success) {
        full_mix = mMuscriptorEngine.takeFinalNotes();
        drop_sparse_rows(full_mix);

        const ScopedLock sl(mStemLock);
        mStemCompleted = full_mix;
        mStemGeneration++;
    } else if (full_mix_outcome == Outcome::Failed) {
        mStemError = "Full mix: " + String(mMuscriptorEngine.getLastErrorMessage());
    }

    mStemFullMixDone = true;

    if (full_mix_outcome != Outcome::Success) {
        mStemCancel = true; // the model cannot run, or the user stopped: no point splitting on
    }

    separation.join();

    if (mStemCancel || full_mix_outcome != Outcome::Success) {
        const bool have_notes = !full_mix.empty();
        mJobOutcome = have_notes ? Outcome::Success
                                 : (full_mix_outcome == Outcome::Failed ? Outcome::Failed : Outcome::Cancelled);
        mJobFinished = true;
        return;
    }

    if (!manifest.has_value()) {
        // The full-mix rows stand on their own: that is plain NeuralNote's result.
        NnLogger::write(NnLogger::Level::Warning, "TranscriptionManager: separation failed: " + separation_error);
        mStemError = "The extra tracks could not be added: "
                     + (separation_error.isEmpty() ? String("the track separation did not finish.") : separation_error);
        mJobOutcome = Outcome::Success;
        mJobFinished = true;
        return;
    }

    mStemBpm = manifest->bpm;

    // 3. Every split track is transcribed and compared with the full-mix rows: what they already
    // show is not repeated, a richer version of a row replaces it, and what they missed is added.
    const std::vector<StemPlan> plans = _planStems(*manifest);
    mStemCount = static_cast<int>(plans.size());

    const int num_workers =
        std::max(1, std::min(static_cast<int>(plans.size()), mJobModelSize == ModelSize::Large ? 2 : 3));

    {
        const ScopedLock sl(mStemLock);
        mStemEngines.clear();

        for (int w = 0; w < num_workers; w++) {
            mStemEngines.push_back(std::make_unique<MuscriptorEngine>());
        }

        mStemDoneCount = 0;
    }

    mStemPhase = 1;

    const std::map<int, std::vector<NoteEvent>> full_mix_rows = NoteMatching::byInstrument(full_mix);

    std::atomic<std::size_t> next_plan {0};
    std::atomic<bool> stop_all {false};
    CriticalSection problems_lock;
    StringArray problems;

    const auto worker = [&](MuscriptorEngine& inEngine) {
        while (!stop_all && !mStemCancel) {
            const std::size_t i = next_plan++;

            if (i >= plans.size()) {
                return;
            }

            const StemPlan& plan = plans[i];

            AudioBuffer<float> audio;
            double sample_rate = 0.0;

            if (!AudioUtils::loadAudioFile(plan.file, audio, sample_rate) || sample_rate <= 0.0) {
                const ScopedLock sl(problems_lock);
                problems.add(plan.name + ": could not read " + plan.file.getFileName());
                continue;
            }

            AudioBuffer<float> mono;
            AudioUtils::resampleBufferToMono(audio, mono, sample_rate, TRANSCRIPTION_SAMPLE_RATE);
            audio.setSize(0, 0);

            if (mono.getNumSamples() >= static_cast<int>(TRANSCRIPTION_SAMPLE_RATE)) {
                NnLogger::write(NnLogger::Level::Info,
                                "TranscriptionManager: track " + String(i + 1) + "/" + String(plans.size()) + ": "
                                    + plan.name);

                inEngine.reset();

                if (mStemCancel) {
                    inEngine.cancel();
                }

                inEngine.setNoteMapper(plan.mapper);

                const Outcome outcome = inEngine.transcribeToMIDI(mJobModelSize,
                                                                  mJobDevice,
                                                                  mono.getReadPointer(0),
                                                                  mono.getNumSamples(),
                                                                  plan.instruments,
                                                                  {},
                                                                  0.0f);

                if (outcome == Outcome::Cancelled) {
                    return;
                }

                if (outcome != Outcome::Success) {
                    const ScopedLock sl(problems_lock);
                    problems.add(plan.name + ": " + String(inEngine.getLastErrorMessage()));
                    stop_all = true;
                    return;
                }

                std::vector<NoteEvent> notes = inEngine.takeFinalNotes();
                drop_sparse_rows(notes);

                const ScopedLock sl(mStemLock);

                if (plan.name == "drums") {
                    // The kit, split by piece, replaces the full mix's single drums row.
                    mStemCompleted.erase(std::remove_if(mStemCompleted.begin(),
                                                        mStemCompleted.end(),
                                                        [](const NoteEvent& inNote) {
                                                            return inNote.program == msl::DRUM_PROGRAM;
                                                        }),
                                         mStemCompleted.end());
                    mStemCompleted.insert(mStemCompleted.end(), notes.begin(), notes.end());
                } else {
                    const NoteMatching::Merge merge = NoteMatching::mergeTrack(notes, full_mix_rows);

                    NnLogger::write(NnLogger::Level::Info,
                                    "TranscriptionManager: " + plan.name + ": " + String(notes.size()) + " notes, "
                                        + (merge.addTrack ? "added" : "already shown")
                                        + (merge.replacedInstruments.empty()
                                               ? String()
                                               : ", replaces " + String(merge.replacedInstruments.size()) + " row(s)"));

                    if (merge.addTrack) {
                        const std::vector<int> replaced = merge.replacedInstruments;
                        mStemCompleted.erase(
                            std::remove_if(mStemCompleted.begin(),
                                           mStemCompleted.end(),
                                           [&replaced](const NoteEvent& inNote) {
                                               return std::find(replaced.begin(), replaced.end(), inNote.program)
                                                      != replaced.end();
                                           }),
                            mStemCompleted.end());
                        mStemCompleted.insert(mStemCompleted.end(), notes.begin(), notes.end());
                    }
                }

                mStemGeneration++;
            }

            const ScopedLock sl(mStemLock);
            mStemDoneCount++;
        }
    };

    std::vector<std::thread> threads;

    for (int w = 1; w < num_workers; w++) {
        threads.emplace_back([&, w] { worker(*mStemEngines[static_cast<std::size_t>(w)]); });
    }

    worker(*mStemEngines[0]);

    for (auto& thread: threads) {
        thread.join();
    }

    {
        const ScopedLock sl(mStemLock);
        mStemEngines.clear();
    }

    if (!problems.isEmpty()) {
        mStemError = problems.joinIntoString("\n");
    }

    // The full-mix rows are there whatever happened to the extra tracks.
    mJobOutcome = Outcome::Success;
    mJobFinished = true;
}

void TranscriptionManager::_drainStemJob()
{
    bool changed = false;

    // The full-mix pass streams like plain NeuralNote: notes from the start of the song onwards.
    if (!mStemFullMixDone.load()) {
        std::string resume_point;
        float resume_progress = 0.0f;
        const std::size_t before = mStemStreaming.size();
        const double horizon_before = mStemHorizon;

        mMuscriptorEngine.drainNewNotes(mStemStreaming, mStemHorizon, resume_point, resume_progress);
        changed = mStemStreaming.size() != before || mStemHorizon > horizon_before;
    }

    {
        const ScopedLock sl(mStemLock);

        // A finished pass or track: its final notes replace whatever was streamed.
        if (mStemGeneration != mStemSeenGeneration) {
            mStemSeenGeneration = mStemGeneration;
            mStemCompletedCopy = mStemCompleted;
            mStemStreaming.clear();
            changed = true;
        }
    }

    if (!changed) {
        return;
    }

    mRawNotes = mStemCompletedCopy;
    mRawNotes.insert(mRawNotes.end(), mStemStreaming.begin(), mStemStreaming.end());

    mFinalizedThrough = mStemFullMixDone.load() ? mProcessor->getSourceAudioManager()->getAudioSampleDuration()
                                                 : mStemHorizon;

    _updateSavedTranscription();
    _updatePostProcessing();
    _repaintPianoRoll();
}

void TranscriptionManager::_cancelStemEngines()
{
    const ScopedLock sl(mStemLock);

    for (const auto& engine: mStemEngines) {
        engine->cancel();
    }
}
