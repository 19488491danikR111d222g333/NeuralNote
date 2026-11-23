//
// Stem mode: splits the song into tracks with the Stems_to_FL Python tools before transcription.
//

#include "StemSeparator.h"

#include <mutex>
#include <string>
#include <thread>

#include "NNFileUtils.h"
#include "NnLogger.h"

namespace
{
// Where Stems_to_FL is on the machine this build is made for. stems_dir.txt overrides it.
const char* const DEFAULT_STEMS_DIR = "G:\\Users\\Danik\\Desktop\\COMFYUIROBOTA\\Stems_to_FL";

const char* const STEMS_DIR_FILE = "stems_dir.txt";
const char* const MANIFEST_FILE = "manifest.json";

// The child prints these, one per line, on stdout.
const char* const PROGRESS_PREFIX = "NNPROGRESS ";
const char* const ERROR_PREFIX = "NNERROR ";

juce::File stemsDirectory()
{
    const juce::String from_env = juce::SystemStats::getEnvironmentVariable("NEURALNOTE_STEMS_DIR", {}).trim();

    if (from_env.isNotEmpty()) {
        return juce::File(from_env);
    }

    const juce::File config = NNFileUtils::getNeuralNoteDirectory().getChildFile(STEMS_DIR_FILE);

    if (config.existsAsFile()) {
        const juce::String first_line = juce::StringArray::fromLines(config.loadFileAsString())[0].trim();

        if (first_line.equalsIgnoreCase("off")) {
            return {};
        }

        if (first_line.isNotEmpty() && juce::File::isAbsolutePath(first_line)) {
            return juce::File(first_line);
        }
    }

    return juce::File(DEFAULT_STEMS_DIR);
}

juce::File pythonIn(const juce::File& inStemsDir)
{
#if JUCE_WINDOWS
    return inStemsDir.getChildFile("env").getChildFile("Scripts").getChildFile("python.exe");
#else
    return inStemsDir.getChildFile("env").getChildFile("bin").getChildFile("python");
#endif
}
} // namespace

std::optional<StemSeparator::Tools> StemSeparator::findTools()
{
    const juce::File dir = stemsDirectory();

    if (dir == juce::File()) {
        NnLogger::write(NnLogger::Level::Info, "StemSeparator: stem mode is off (stems_dir.txt)");
        return std::nullopt;
    }

    Tools tools {pythonIn(dir), dir.getChildFile("stems_to_fl.py")};

    if (!tools.python.existsAsFile() || !tools.script.existsAsFile()) {
        NnLogger::write(NnLogger::Level::Info,
                        "StemSeparator: Stems_to_FL not found in " + dir.getFullPathName()
                            + ", transcribing the full mix");
        return std::nullopt;
    }

    return tools;
}

juce::File StemSeparator::outputFolderFor(const juce::File& inSource)
{
    // A working copy in NeuralNote's own folder, deleted once the notes are made: nothing is left
    // next to the song.
    return NNFileUtils::getNeuralNoteDirectory().getChildFile("Stems").getChildFile(
        inSource.getFileNameWithoutExtension());
}

std::optional<StemSeparator::Manifest> StemSeparator::readManifest(const juce::File& inFolder)
{
    const juce::File file = inFolder.getChildFile(MANIFEST_FILE);

    if (!file.existsAsFile()) {
        return std::nullopt;
    }

    const juce::var parsed = juce::JSON::parse(file.loadFileAsString());
    const juce::DynamicObject* root = parsed.getDynamicObject();

    if (root == nullptr) {
        return std::nullopt;
    }

    Manifest manifest;
    manifest.folder = inFolder;
    manifest.bpm = static_cast<double>(root->getProperty("bpm"));
    manifest.key = root->getProperty("key").toString();

    if (const juce::DynamicObject* stems = root->getProperty("stems").getDynamicObject()) {
        for (const auto& property: stems->getProperties()) {
            // Paths are relative to the folder, so a moved or restored _FL folder still reads.
            const juce::File stem = inFolder.getChildFile(property.value.toString());

            if (stem.existsAsFile()) {
                manifest.stems[property.name.toString()] = stem;
            }
        }
    }

    if (manifest.stems.empty()) {
        return std::nullopt;
    }

    return manifest;
}

std::optional<StemSeparator::Manifest> StemSeparator::run(const Tools& inTools,
                                                          const juce::File& inSource,
                                                          const juce::File& inOutFolder,
                                                          const std::function<bool()>& inShouldCancel,
                                                          const std::function<void(float)>& inOnProgress,
                                                          juce::String& outError)
{
    outError.clear();

    if (!NNFileUtils::ensureDirectoryExists(inOutFolder)) {
        outError = "Could not create " + inOutFolder.getFullPathName();
        return std::nullopt;
    }

    const juce::StringArray command {inTools.python.getFullPathName(),
                                     "-u",
                                     inTools.script.getFullPathName(),
                                     inSource.getFullPathName(),
                                     "--nn-out",
                                     inOutFolder.getFullPathName()};

    NnLogger::write(NnLogger::Level::Info, "StemSeparator: " + command.joinIntoString(" "));

    juce::ChildProcess child;

    if (!child.start(command, juce::ChildProcess::wantStdOut | juce::ChildProcess::wantStdErr)) {
        outError = "Could not start " + inTools.python.getFullPathName();
        return std::nullopt;
    }

    const double start_ms = juce::Time::getMillisecondCounterHiRes();
    juce::String tail; // last lines, for the error message and the log

    const auto handle_line = [&](const juce::String& inLine) {
        const juce::String line = inLine.trimEnd();

        if (line.startsWith(PROGRESS_PREFIX)) {
            const float fraction = line.fromFirstOccurrenceOf(PROGRESS_PREFIX, false, false).getFloatValue();
            inOnProgress(juce::jlimit(0.0f, 1.0f, fraction));
        } else if (line.startsWith(ERROR_PREFIX)) {
            outError = line.fromFirstOccurrenceOf(ERROR_PREFIX, false, false);
        } else if (line.isNotEmpty() && !line.containsChar('\r')) {
            tail = (tail + "\n" + line).getLastCharacters(2000);
        }
    };

    // ChildProcess::readProcessOutput blocks until its buffer is full or the process ends, so it
    // runs on a thread of its own and this one stays free to notice a cancellation and kill the
    // child, which is also what unblocks the reader.
    std::mutex output_mutex;
    std::string output;
    std::atomic<bool> reader_done {false};

    std::thread reader([&] {
        char byte = 0;

        while (child.readProcessOutput(&byte, 1) == 1) {
            const std::lock_guard<std::mutex> lock(output_mutex);
            output.push_back(byte);
        }

        reader_done = true;
    });

    juce::String pending;
    bool cancelled = false;

    const auto take_output = [&] {
        std::string chunk;
        {
            const std::lock_guard<std::mutex> lock(output_mutex);
            chunk.swap(output);
        }

        // tqdm redraws its bar with carriage returns; a line is whatever ends in \n.
        pending += juce::String::fromUTF8(chunk.data(), static_cast<int>(chunk.size()));

        while (pending.containsChar('\n')) {
            handle_line(pending.upToFirstOccurrenceOf("\n", false, false));
            pending = pending.fromFirstOccurrenceOf("\n", false, false);
        }
    };

    while (!reader_done) {
        if (!cancelled && inShouldCancel()) {
            cancelled = true;
            child.kill();
        }

        take_output();
        juce::Thread::sleep(50);
    }

    reader.join();
    take_output();

    if (cancelled) {
        NnLogger::write(NnLogger::Level::Info, "StemSeparator: cancelled, child process killed");
        return std::nullopt;
    }

    handle_line(pending);

    const juce::uint32 exit_code = child.getExitCode();
    const int seconds = juce::roundToInt((juce::Time::getMillisecondCounterHiRes() - start_ms) / 1000.0);

    NnLogger::write(NnLogger::Level::Info,
                    "StemSeparator: finished in " + juce::String(seconds) + " s, exit code " + juce::String(exit_code)
                        + "\n" + tail);

    std::optional<Manifest> manifest = readManifest(inOutFolder);

    if (exit_code != 0 || !manifest.has_value()) {
        if (outError.isEmpty()) {
            outError = "The track separation did not finish (exit code " + juce::String(exit_code)
                       + "). See log.txt in " + inOutFolder.getFullPathName();
        }

        return std::nullopt;
    }

    return manifest;
}
