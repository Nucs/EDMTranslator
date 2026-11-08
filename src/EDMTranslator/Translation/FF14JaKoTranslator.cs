using EDMTranslator.Models;
using EDMTranslator.Tokenization;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace EDMTranslator.Translation
{
    /// <summary>
    /// Japanese-to-Korean translator on <c>sappho192/ffxiv-ja-ko-translator</c> (bert-base-japanese-v2 encoder, KoGPT2
    /// decoder, fine-tuned on Final Fantasy XIV text). Decodes greedily up to 50 tokens; <c>&lt;/s&gt;</c> (1) both starts
    /// and ends the output.
    /// </summary>
    public class FF14JaKoTranslator : EncoderDecoderTranslator
    {
        /// <summary>The encoder (<c>encoder_model.onnx</c>); an <see cref="NDInferenceSession"/>, whose release is deferred until a run in flight returns.</summary>
        private readonly NDInferenceSession encoderSession;

        /// <summary>The merged decoder (<c>decoder_model_merged.onnx</c>).</summary>
        private readonly NDInferenceSession decoderSession;

        /// <summary>Opens the model's two ONNX files with ONNX Runtime's default options; loading takes a few seconds and about 1.5 GB of memory.</summary>
        /// <param name="tokenizer">The tokenizer: bert-base-japanese-v2 for the source, the KoGPT2 <c>tokenizer.json</c> for the output.</param>
        /// <param name="modelDirPath">The folder holding <c>encoder_model.onnx</c> and <c>decoder_model_merged.onnx</c>.</param>
        /// <exception cref="OnnxRuntimeException">A model file is missing or is not a valid ONNX model.</exception>
        public FF14JaKoTranslator(BertJa2GPTTokenizer tokenizer, string modelDirPath)
            : this(tokenizer, modelDirPath, sessionOptions: null)
        {
        }

        /// <summary>Opens the model's two ONNX files with the given ONNX Runtime options.</summary>
        /// <remarks>
        /// The options are how to tune ONNX Runtime, which spends nearly all of a translation's time. The one that matters most
        /// on a desktop CPU is <see cref="SessionOptions.IntraOpNumThreads"/>: the default uses every physical core, and on a
        /// hybrid CPU (performance + efficiency cores) a decoder step then waits for its slowest thread - on an i9-13900K
        /// under load, the median step took 30-81 ms with the default against ~20 ms with 8 or 16 threads.
        /// </remarks>
        /// <param name="tokenizer">The tokenizer: bert-base-japanese-v2 for the source, the KoGPT2 <c>tokenizer.json</c> for the output.</param>
        /// <param name="modelDirPath">The folder holding <c>encoder_model.onnx</c> and <c>decoder_model_merged.onnx</c>.</param>
        /// <param name="sessionOptions">Options for both sessions, or <c>null</c> for ONNX Runtime's defaults with error-level
        /// logging (what the two-argument constructor uses). They stay the caller's: the sessions copy them, so they may be
        /// disposed as soon as this returns.</param>
        /// <exception cref="OnnxRuntimeException">A model file is missing or is not a valid ONNX model, or the options are invalid.</exception>
        public FF14JaKoTranslator(BertJa2GPTTokenizer tokenizer, string modelDirPath, SessionOptions? sessionOptions) : base(
            modelName: "ffxiv-ja-ko-translator",
            modelHubName: "sappho192/ffxiv-ja-ko-translator",
            tokenizer: tokenizer)
        {
            string encoderModelPath = Path.Combine(modelDirPath, "encoder_model.onnx");
            string decoderModelPath = Path.Combine(modelDirPath, "decoder_model_merged.onnx");
            // The sessions copy their options when they are created, so the default options' native handle can go right
            // after (using on null does nothing: caller-supplied options are the caller's to dispose).
            using var defaultOptions = sessionOptions is null
                ? new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR }
                : null;
            encoderSession = new(encoderModelPath, sessionOptions ?? defaultOptions!);
            decoderSession = new(decoderModelPath, sessionOptions ?? defaultOptions!);
        }

        /// <summary>Translates one Japanese sentence into Korean.</summary>
        /// <param name="sentence">The Japanese text.</param>
        /// <returns>The Korean translation.</returns>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime failed to run a model.</exception>
        public override string Translate(string sentence)
        {
            (var inputIds, var attentionMask) = Tokenizer.Encode(sentence);
            //Console.WriteLine($"Input tokens: {string.Join(", ", inputIds)}");

            // inputIds to NDArray: one (1, S) row each, built straight from the jagged array. They stay int32: the bridge
            // casts them to the encoder's declared int64 as it feeds them (a "safe" cast, so nothing is lost).
            using NDArray ndInputIds = np.array(new[] { inputIds });
            using NDArray ndAttentionMask = np.array(new[] { attentionMask });

            var encoderInput = new Dictionary<string, NDArray>
            {
                ["input_ids"] = ndInputIds,
                ["attention_mask"] = ndAttentionMask,
            };

            // zeroCopyOutputs: the hidden states stay in ONNX Runtime's buffer, viewed as an NDArray that owns it, and
            // every decoder step reads them from there.
            using NDArray encoderHiddenStates = encoderSession.Run(encoderInput, new[] { "last_hidden_state" }, zeroCopyOutputs: true)["last_hidden_state"];

            var generatedText = GreedySearch(encoderHiddenStates, decoderSession, 1, 1);
            return generatedText;
        }

        /// <summary>Releases the encoder and decoder sessions (see <see cref="EncoderDecoderTranslator.Dispose()"/>).</summary>
        /// <param name="disposing"><c>true</c> from <see cref="EncoderDecoderTranslator.Dispose()"/>.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // NDInferenceSession.Dispose is idempotent and defers the native release to a run in flight on another thread.
                encoderSession.Dispose();
                decoderSession.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
