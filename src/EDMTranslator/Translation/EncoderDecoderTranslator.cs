using System.Runtime.CompilerServices;
using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace EDMTranslator.Models
{
    /// <summary>
    /// Base of the translators that pair a BERT encoder with a GPT-2 decoder exported by Hugging Face Optimum
    /// (<c>encoder_model.onnx</c> + <c>decoder_model_merged.onnx</c>) and decode greedily.
    /// </summary>
    /// <remarks>
    /// <para>The search (<see cref="GreedySearch(NDArray, InferenceSession, long, long, int)"/>) runs on NumSharp arrays that
    /// share memory with ONNX Runtime through <c>NumSharp.Interop.OnnxRuntime</c>: the encoder's hidden states, each step's
    /// logits and the key/value cache stay in the buffers ONNX Runtime wrote, viewed as <see cref="NDArray"/>s and fed back
    /// to the next step without a copy, and the next token is <c>np.argmax</c> of the last position's logits. Each step's
    /// cache is released as soon as the next step has read it, so a long translation holds one step's cache, not all of them.</para>
    /// <para>The members built on <see cref="NamedOnnxValue"/> are the earlier surface, kept for subclasses:
    /// <see cref="GreedySearch(List{NamedOnnxValue}, InferenceSession, long, long, int)"/> now runs the NumSharp search, and
    /// <see cref="ForwardOnnx(InferenceSession, List{NamedOnnxValue})"/>, <see cref="CombineTuples"/>,
    /// <see cref="Softmax"/>, <see cref="ArgMax"/> and <see cref="InitKeyValues"/> stay callable.</para>
    /// </remarks>
    public abstract class EncoderDecoderTranslator : Translator
    {
        /// <summary>
        /// Each decoder session's input/output layout, read once from its metadata on the first search and kept for the
        /// session's lifetime (the table holds the session weakly, so a released session takes its entry with it).
        /// </summary>
        private static readonly ConditionalWeakTable<InferenceSession, DecoderLayout> Layouts = new();

        /// <summary>Describes the model; the subclass opens the encoder and decoder sessions.</summary>
        /// <param name="modelName">The model's short name, e.g. <c>jesc-ja-en-translator</c>.</param>
        /// <param name="modelHubName">The model's Hugging Face id, <c>owner/name</c>; also builds <see cref="ModelInfo.ModelUri"/>.</param>
        /// <param name="tokenizer">Encodes the source sentence and decodes the generated tokens.</param>
        public EncoderDecoderTranslator(string modelName, string modelHubName, ITokenizer tokenizer) :
            base(modelInfo: new ModelInfo(
                modelName: modelName,
                modelHubName: modelHubName,
                modelType: ModelType.EncoderDecoder,
                modelUri: $"https://huggingface.co/{modelHubName}"),
            tokenizer)
        {

        }

        /// <summary>
        /// A zero-filled (1, 12, <paramref name="inputIdLength"/>, 64) float tensor: the placeholder cache entry of a
        /// 12-head, 64-wide decoder layer that <see cref="ForwardOnnx(InferenceSession, List{NamedOnnxValue})"/> feeds on the
        /// first step. The NumSharp search does not use it (it builds one shared placeholder per search).
        /// </summary>
        /// <param name="inputIdLength">The placeholder's sequence length (the first step uses 1).</param>
        /// <returns>A new <see cref="DenseTensor{T}"/> over its own managed array; the caller owns it.</returns>
        /// <exception cref="ArgumentException"><paramref name="inputIdLength"/> is negative.</exception>
        protected static Tensor<float> InitKeyValues(int inputIdLength)
        {
            // NumSharp 0.80 keeps NumPy's one signature, np.zeros(shape, dtype); the generic params-int[] overload is gone.
            // ToDenseTensor copies once, straight into the tensor's array (EDMTranslator 1.3.1 went NDArray -> float[,,,] -> DenseTensor).
            using NDArray zeros = np.zeros(new Shape(1, 12, inputIdLength, 64), np.float32);
            return zeros.ToDenseTensor<float>();
        }

        /// <summary>
        /// Greedy search on the merged decoder: starts from <paramref name="bosTokenId"/>, takes the most likely token each
        /// step until <paramref name="eosTokenId"/> or <paramref name="maxLength"/> tokens, and decodes what it generated.
        /// </summary>
        /// <remarks>
        /// <para>Equal to transformers' <c>generate(num_beams=1, do_sample=False)</c> on the same encoder output: the token is
        /// the argmax of the last position's logits (the first one on ties), with no logits processor. The softmax
        /// EDMTranslator 1.3.1 applied first is monotonic, so it never changed the choice and is not computed.</para>
        /// <para>Every tensor stays where ONNX Runtime put it: <c>session.Run(..., zeroCopyOutputs: true)</c> returns the logits
        /// and the <c>present.*</c> cache as NDArray views that own ORT's buffers, the cache views are the next step's
        /// <c>past_key_values.*</c> inputs (fed zero-copy, as is <paramref name="encoderHiddenStates"/>), and a step's cache is
        /// disposed once the next run has returned. The first step feeds one shared zero placeholder for every cache input:
        /// the merged decoder's no-cache branch (<c>use_cache_branch = false</c>) does not read it, but ONNX requires a value.</para>
        /// <para>The layout - which inputs are cache entries, the <c>present.*</c> output that continues each one, the cache's
        /// head count and width - is read from <paramref name="decoderSession"/>'s metadata, so any Optimum merged decoder
        /// with this input naming works, whatever its layer count.</para>
        /// </remarks>
        /// <param name="encoderHiddenStates">The encoder's <c>last_hidden_state</c>, (1, source length, hidden) float32. Read by
        /// every step; it is neither modified nor disposed.</param>
        /// <param name="decoderSession">The decoder session (<c>decoder_model_merged.onnx</c>).</param>
        /// <param name="bosTokenId">The decoder start token, fed on the first step and not part of the result.</param>
        /// <param name="eosTokenId">The token that ends the search; it is part of the decoded sequence when reached.</param>
        /// <param name="maxLength">The most tokens to generate (EOS included); 0 or less generates nothing.</param>
        /// <returns>The generated tokens decoded by <see cref="Translator.Tokenizer"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="encoderHiddenStates"/> or <paramref name="decoderSession"/> is null.</exception>
        /// <exception cref="NotSupportedException">The decoder is not an Optimum merged decoder: it lacks <c>input_ids</c>,
        /// <c>encoder_hidden_states</c>, <c>use_cache_branch</c> or <c>past_key_values.*</c> inputs, a cache input has no
        /// <c>present.*</c> output, or a cache input is not rank 4.</exception>
        /// <exception cref="ArgumentException"><paramref name="encoderHiddenStates"/> contradicts the declared shape of
        /// <c>encoder_hidden_states</c>.</exception>
        /// <exception cref="InvalidCastException"><paramref name="encoderHiddenStates"/> holds a dtype that does not cast safely to the declared float32.</exception>
        /// <exception cref="ObjectDisposedException"><paramref name="encoderHiddenStates"/> or the session was disposed.</exception>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime failed to run the model.</exception>
        protected string GreedySearch(NDArray encoderHiddenStates, InferenceSession decoderSession,
            long bosTokenId, long eosTokenId, int maxLength = 50)
        {
            ArgumentNullException.ThrowIfNull(encoderHiddenStates);
            ArgumentNullException.ThrowIfNull(decoderSession);
            DecoderLayout layout = Layouts.GetValue(decoderSession, DecoderLayout.Read);

            // Initialize the input for the decoder with the BOS token ID: a (1, 1) int64 array rewritten in place each step.
            // The bridge feeds it zero-copy during a run, and nothing reads it between runs.
            using NDArray inputIds = np.array(new long[,] { { bosTokenId } });
            // false selects the merged decoder's no-cache branch for the first step, true the cache branch from then on.
            using NDArray useCacheBranch = np.array(new[] { false });
            // The first step's past: the no-cache branch never reads it, but ONNX needs a value for every input, so one zero
            // (1, heads, 1, width) array serves all of them (EDMTranslator 1.3.1 built one per input, each copied twice).
            using NDArray emptyPast = np.zeros(new Shape(1, layout.Heads, 1, layout.HeadDim), np.float32);

            var feeds = new Dictionary<string, NDArray>(layout.PastNames.Length + 3)
            {
                ["input_ids"] = inputIds,
                ["encoder_hidden_states"] = encoderHiddenStates,
                ["use_cache_branch"] = useCacheBranch,
            };
            foreach (string name in layout.PastNames)
                feeds[name] = emptyPast;

            // Initialize the list to store the generated tokens
            List<long> generatedTokens = new();
            IReadOnlyDictionary<string, NDArray>? step = null;
            try
            {
                // Greedy search loop
                for (int i = 0; i < maxLength; i++)
                {
                    // Run the decoder model. zeroCopyOutputs: the logits and the cache come back as NDArray views of the
                    // buffers ONNX Runtime wrote, each owning its OrtValue.
                    IReadOnlyDictionary<string, NDArray> outputs = decoderSession.Run(feeds, layout.OutputNames, zeroCopyOutputs: true);
                    // The previous step's cache was this run's input; the run has returned, so its buffers go back to ONNX
                    // Runtime now rather than whenever a collection finalizes them.
                    DisposeAll(step);
                    step = outputs;

                    // Get the logits of the last position from the decoder results and take the most likely token:
                    // np.argmax over a view of ORT's buffer (the first maximum on ties), no softmax and no copy.
                    long nextTokenId;
                    using (NDArray lastLogits = step[layout.LogitsName]["0, -1"])
                        nextTokenId = np.argmax(lastLogits);

                    // Append the token to the list
                    generatedTokens.Add(nextTokenId);

                    // Check if EOS token is generated
                    if (nextTokenId == eosTokenId)
                        break;

                    // Prepare the input for the next iteration: the token, the cache branch, and this step's cache as the
                    // next step's past - every layer's present.* output, fed back without a copy.
                    inputIds.SetInt64(nextTokenId, 0, 0);
                    useCacheBranch.SetBoolean(true, 0);
                    for (int k = 0; k < layout.PastNames.Length; k++)
                        feeds[layout.PastNames[k]] = step[layout.PresentNames[k]];
                }
            }
            finally
            {
                // Also on an exception: the last outputs (or the previous step's, if the run threw) are released here.
                DisposeAll(step);
            }

            // Decode the generated tokens into text
            var generatedText = Tokenizer.Decode(generatedTokens.Select(item => (uint)item).ToArray());
            return generatedText;
        }

        /// <summary>
        /// Greedy search over decoder inputs built as <see cref="NamedOnnxValue"/>s (the EDMTranslator 1.3.1 form); runs
        /// <see cref="GreedySearch(NDArray, InferenceSession, long, long, int)"/> on their <c>encoder_hidden_states</c>.
        /// </summary>
        /// <remarks>Only the <c>encoder_hidden_states</c> entry is read; <c>input_ids</c>, <c>use_cache_branch</c> and any
        /// <c>past_key_values.*</c> entries are the search's own and are ignored, and <paramref name="decoderInput"/> is no
        /// longer modified. A dense tensor (what ONNX Runtime's legacy <c>Run</c> returns) is viewed in place, any other copied.</remarks>
        /// <param name="decoderInput">The decoder inputs; must hold <c>encoder_hidden_states</c> as a float tensor.</param>
        /// <param name="decoderSession">The decoder session (<c>decoder_model_merged.onnx</c>).</param>
        /// <param name="bosTokenId">The decoder start token.</param>
        /// <param name="eosTokenId">The token that ends the search.</param>
        /// <param name="maxLength">The most tokens to generate (EOS included).</param>
        /// <returns>The generated tokens decoded by <see cref="Translator.Tokenizer"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="decoderInput"/> or <paramref name="decoderSession"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="decoderInput"/> has no <c>encoder_hidden_states</c> entry.</exception>
        /// <exception cref="InvalidCastException">The <c>encoder_hidden_states</c> entry is not a float tensor.</exception>
        /// <exception cref="NotSupportedException">The decoder is not an Optimum merged decoder (see the other overload).</exception>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime failed to run the model.</exception>
        protected string GreedySearch(List<NamedOnnxValue> decoderInput, InferenceSession decoderSession,
            long bosTokenId, long eosTokenId, int maxLength = 50)
        {
            ArgumentNullException.ThrowIfNull(decoderInput);
            NamedOnnxValue hidden = decoderInput.Find(input => input.Name.Equals("encoder_hidden_states"))
                ?? throw new ArgumentException("The decoder inputs have no 'encoder_hidden_states' entry.", nameof(decoderInput));
            using NDArray encoderHiddenStates = hidden.AsTensor<float>() is DenseTensor<float> dense ? dense.AsNDArray() : hidden.ToNDArray();
            return GreedySearch(encoderHiddenStates, decoderSession, bosTokenId, eosTokenId, maxLength);
        }

        /// <summary>
        /// One decoder step on the no-cache branch (the first step, <c>use_cache_branch = false</c>): adds a zero placeholder
        /// for each of the 24 cache inputs of a 12-layer decoder, runs the decoder, and returns its results with the cache the
        /// next step feeds.
        /// </summary>
        /// <param name="session">The decoder session (<c>decoder_model_merged.onnx</c>).</param>
        /// <param name="input_data">The decoder inputs (<c>input_ids</c>, <c>encoder_hidden_states</c>, <c>use_cache_branch</c>);
        /// the 24 <c>past_key_values.*</c> placeholders are appended to it.</param>
        /// <returns>The run's results (logits first; the caller owns and disposes them) and the cache for the next step,
        /// grouped as <see cref="ForwardOnnx(InferenceSession, List{NamedOnnxValue}, IEnumerable{IEnumerable{DisposableNamedOnnxValue}})"/> expects.</returns>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime rejected the inputs or failed to run the model.</exception>
        protected (IDisposableReadOnlyCollection<DisposableNamedOnnxValue>, IEnumerable<IEnumerable<DisposableNamedOnnxValue>>) ForwardOnnx(
            InferenceSession session, List<NamedOnnxValue> input_data)
        {
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.0.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.0.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.1.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.1.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.2.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.2.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.3.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.3.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.4.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.4.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.5.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.5.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.6.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.6.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.7.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.7.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.8.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.8.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.9.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.9.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.10.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.10.value", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.11.key", InitKeyValues(1)));
            input_data.Add(NamedOnnxValue.CreateFromTensor("past_key_values.11.value", InitKeyValues(1)));

            var results = session.Run(input_data);

            var outPastKeyValues = results.Skip(1).ToList();
            List<List<DisposableNamedOnnxValue>> newPastKeyValues;

            // Similar to the Python code, depending on the value of use_cache_branch, pack the out_past_key_values into groups of num_pkv units
            // ((DenseTensor<bool>)input_data[2].Value)[0]
            // Pack them to 4 units
            newPastKeyValues = outPastKeyValues.Select((_, index) => outPastKeyValues.Skip(index * 4).Take(4).ToList()).ToList();
            // Make a list using the 0~5th units of newPastKeyValues
            newPastKeyValues = newPastKeyValues.Take(6).ToList();

            return (results, newPastKeyValues);
        }

        /// <summary>
        /// One decoder step on the cache branch (<c>use_cache_branch = true</c>): feeds the previous step's cache, runs the
        /// decoder, and returns its results with the cache the next step feeds.
        /// </summary>
        /// <param name="session">The decoder session (<c>decoder_model_merged.onnx</c>).</param>
        /// <param name="input_data">The decoder inputs; their <c>use_cache_branch</c> and <c>past_key_values.*</c> entries are replaced in place.</param>
        /// <param name="packedPastKeyValues">The previous step's cache, as this method or the first-step overload returned it.</param>
        /// <returns>The run's results (logits first; the caller owns and disposes them) and the cache for the next step.</returns>
        /// <exception cref="NotSupportedException">The decoder has no <c>past_key_values.0.*</c> input (see <see cref="PastValuesPerLayer"/>).</exception>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime rejected the inputs or failed to run the model.</exception>
        protected (IDisposableReadOnlyCollection<DisposableNamedOnnxValue>, IEnumerable<IEnumerable<DisposableNamedOnnxValue>>) ForwardOnnx(
    InferenceSession session, List<NamedOnnxValue> input_data, IEnumerable<IEnumerable<DisposableNamedOnnxValue>> packedPastKeyValues)
        {
            List<DisposableNamedOnnxValue> past_key_values;
            // Unpack the packedPastKeyValues to past_key_values
            past_key_values = packedPastKeyValues.SelectMany(item => item).ToList();

            var singleBoolArray = new bool[] { true };
            var useCacheBranch = np.array(singleBoolArray).ToMuliDimArray<bool>().ToTensor<bool>();
            input_data.Find(input => input.Name.Equals("use_cache_branch")).Value = useCacheBranch;
            input_data.Find(input => input.Name.Equals("past_key_values.0.key")).Value = past_key_values[0].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.0.value")).Value = past_key_values[1].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.1.key")).Value = past_key_values[2].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.1.value")).Value = past_key_values[3].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.2.key")).Value = past_key_values[4].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.2.value")).Value = past_key_values[5].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.3.key")).Value = past_key_values[6].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.3.value")).Value = past_key_values[7].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.4.key")).Value = past_key_values[8].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.4.value")).Value = past_key_values[9].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.5.key")).Value = past_key_values[10].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.5.value")).Value = past_key_values[11].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.6.key")).Value = past_key_values[12].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.6.value")).Value = past_key_values[13].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.7.key")).Value = past_key_values[14].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.7.value")).Value = past_key_values[15].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.8.key")).Value = past_key_values[16].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.8.value")).Value = past_key_values[17].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.9.key")).Value = past_key_values[18].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.9.value")).Value = past_key_values[19].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.10.key")).Value = past_key_values[20].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.10.value")).Value = past_key_values[21].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.11.key")).Value = past_key_values[22].Value;
            input_data.Find(input => input.Name.Equals("past_key_values.11.value")).Value = past_key_values[23].Value;

            var results = session.Run(input_data);

            var outPastKeyValues = results.Skip(1).ToList();
            IEnumerable<IEnumerable<DisposableNamedOnnxValue>> newPastKeyValues;

            // Combine the first two units of out_past_key_values with the last two units of past_key_values.
            // The unit is one decoder layer's cache entries, and how many a layer has depends on the export: Optimum's
            // seq2seq decoders (T5, BART) have four - the self-attention key/value the step extends, then the cross-attention
            // key/value that never change and are carried over from the input - while the GPT-2 decoders of these models have
            // two, both self-attention, so every entry must come from this step's outputs. A fixed unit of four took layers
            // 1, 3, 5, 7, 9 and 11 from the input every step, so they kept the first step's cache and never saw a generated token.
            var unpackedNewPastKeyValues = CombineTuples(outPastKeyValues, past_key_values, PastValuesPerLayer(session));
            // Pack them to 4 units
            newPastKeyValues = unpackedNewPastKeyValues.Select((_, index) => unpackedNewPastKeyValues.Skip(index * 4).Take(4));
            // Make a list using the 0~5th units of newPastKeyValues
            newPastKeyValues = newPastKeyValues.Take(6);

            return (results, newPastKeyValues);
        }

        /// <summary>
        /// The number of cache tensors one decoder layer takes, read from the session's input names: the inputs of layer 0
        /// (<c>past_key_values.0.*</c>). Two for a GPT-2 decoder (self-attention key and value), four for an Optimum seq2seq
        /// decoder that also caches cross-attention (<c>past_key_values.0.decoder.key</c> ... <c>past_key_values.0.encoder.value</c>).
        /// </summary>
        /// <param name="decoderSession">The decoder session (<c>decoder_model_merged.onnx</c>).</param>
        /// <returns>The per-layer count, the unit <see cref="CombineTuples"/> groups by.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="decoderSession"/> is null.</exception>
        /// <exception cref="NotSupportedException">The model has no <c>past_key_values.0.*</c> input, so it is not a decoder with a key/value cache.</exception>
        protected static int PastValuesPerLayer(InferenceSession decoderSession)
        {
            ArgumentNullException.ThrowIfNull(decoderSession);
            int count = decoderSession.InputMetadata.Keys.Count(name => name.StartsWith("past_key_values.0.", StringComparison.Ordinal));
            if (count == 0)
                throw new NotSupportedException("The decoder model has no 'past_key_values.0.*' input; a decoder with a key/value cache is expected.");
            return count;
        }

        /// <summary>
        /// Rebuilds the cache after a cache-branch step, one layer of <paramref name="numPkvs"/> entries at a time: the
        /// layer's first two entries (self-attention key and value) come from the step's outputs, the rest (cross-attention
        /// key and value, when the export caches them) are carried over from the step's inputs.
        /// </summary>
        /// <param name="outPastKeyValues">The step's cache outputs, layer by layer.</param>
        /// <param name="pastKeyValues">The cache the step was fed, in the same order.</param>
        /// <param name="numPkvs">Entries per layer (<see cref="PastValuesPerLayer"/>): 2 takes everything from the outputs.</param>
        /// <returns>The next step's cache, flattened in layer order.</returns>
        /// <exception cref="ArgumentException">The lists are shorter than <paramref name="numPkvs"/> implies, or <paramref name="numPkvs"/> is below 2.</exception>
        protected static List<DisposableNamedOnnxValue> CombineTuples(List<DisposableNamedOnnxValue> outPastKeyValues, List<DisposableNamedOnnxValue> pastKeyValues, int numPkvs)
        {
            List<DisposableNamedOnnxValue> newOutPastKeyValues = new();
            for (int i = 0; i < outPastKeyValues.Count; i += numPkvs)
            {
                var temp = outPastKeyValues.GetRange(i, 2);
                temp.AddRange(pastKeyValues.GetRange(i + 2, numPkvs - 2));
                newOutPastKeyValues.AddRange(temp);
            }
            return newOutPastKeyValues;
        }

        /// <summary>
        /// The softmax of every element of <paramref name="logits"/>, flattened: max-shifted, so the largest logit maps to
        /// exp(0) and nothing overflows. Computed by NumSharp in float32 (<c>Postprocess.Softmax</c>).
        /// </summary>
        /// <remarks>Agrees with EDMTranslator 1.3.1's LINQ version within float32 rounding (that one exponentiated in double and summed
        /// in double before dividing). A NaN logit makes every probability NaN, as NumPy does.</remarks>
        /// <param name="logits">The logits; a dense tensor is read in place, any other copied once.</param>
        /// <returns>A new array of probabilities that sums to 1.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="logits"/> is null.</exception>
        protected static float[] Softmax(Tensor<float> logits)
        {
            using NDArray flat = Flatten(logits);
            // The same four steps as the LINQ version, vectorized: find the maximum value in the logits, subtract it from
            // each logit to avoid overflow during the exponential operation, calculate the sum of the exponentials, and
            // divide each exponential by the sum to get the softmax probabilities.
            using NDArray probabilities = Postprocess.Softmax(flat);
            return probabilities.ToArray<float>();
        }

        /// <summary>
        /// The index of the largest element of <paramref name="probabilities"/> (flattened, the first one on ties), or with
        /// <paramref name="getSecond"/> the first index holding the second value of a descending sort - which is the maximum
        /// again when the maximum occurs twice.
        /// </summary>
        /// <remarks>NumSharp computes both (<c>np.argmax</c>; <c>np.partition</c> for the second value). For input without NaN
        /// the result equals EDMTranslator 1.3.1's LINQ version; with NaN it follows NumPy (<c>np.argmax</c> returns the first NaN).</remarks>
        /// <param name="probabilities">The scores; a dense tensor is read in place, any other copied once.</param>
        /// <param name="getSecond"><c>true</c> for the runner-up's index instead of the maximum's.</param>
        /// <returns>The flat index.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="probabilities"/> is null.</exception>
        /// <exception cref="InvalidOperationException">The tensor is empty, or has one element and <paramref name="getSecond"/> is set.</exception>
        protected static int ArgMax(Tensor<float> probabilities, bool getSecond = false)
        {
            using NDArray flat = Flatten(probabilities);
            long size = flat.size;
            if (size == 0 || (getSecond && size < 2))
                throw new InvalidOperationException("Sequence contains no elements");
            if (getSecond)
            {
                // Find the index of the second maximum value in the probabilities
                using NDArray partitioned = np.partition(flat, (int)(size - 2));
                float second = partitioned.GetSingle(size - 2);
                using NDArray isSecond = flat == second;
                return (int)np.argmax(isSecond);
            }
            // Find the index of the maximum value in the probabilities
            return (int)np.argmax(flat);
        }

        /// <summary>A 1-D NDArray over every element of <paramref name="tensor"/> in row-major order: a view of a dense tensor's own memory, else a copy.</summary>
        /// <param name="tensor">The tensor to read.</param>
        /// <returns>An array the caller disposes; disposing a view leaves the tensor intact.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        private static NDArray Flatten(Tensor<float> tensor)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            using NDArray nd = tensor is DenseTensor<float> dense ? dense.AsNDArray() : tensor.ToNDArray();
            return np.ravel(nd);
        }

        /// <summary>Disposes every array of a run's outputs, releasing the ONNX Runtime buffers the views own.</summary>
        /// <param name="arrays">The outputs, or <c>null</c> (nothing to do).</param>
        private static void DisposeAll(IReadOnlyDictionary<string, NDArray>? arrays)
        {
            if (arrays is null)
                return;
            foreach (NDArray array in arrays.Values)
                array.Dispose();
        }

        /// <summary>
        /// Where a merged decoder keeps its cache: the <c>past_key_values.*</c> inputs in model order, the <c>present.*</c>
        /// output that continues each one, the outputs a step fetches, and the head count and width of a cache entry.
        /// </summary>
        private sealed class DecoderLayout
        {
            /// <summary>The prefix of the cache inputs (Optimum's naming).</summary>
            private const string PastPrefix = "past_key_values.";

            /// <summary>The prefix of the cache outputs; <c>present.N.key</c> continues <c>past_key_values.N.key</c>.</summary>
            private const string PresentPrefix = "present.";

            /// <summary>The cache inputs, in the model's input order.</summary>
            public string[] PastNames { get; }

            /// <summary>The cache output that continues each of <see cref="PastNames"/>, index for index.</summary>
            public string[] PresentNames { get; }

            /// <summary>The logits output (<c>logits</c>, or the first output of a model without one of that name).</summary>
            public string LogitsName { get; }

            /// <summary>What each step fetches: the logits, then <see cref="PresentNames"/>.</summary>
            public string[] OutputNames { get; }

            /// <summary>The attention heads of a cache entry (dimension 1), or 1 when the model leaves it symbolic.</summary>
            public int Heads { get; }

            /// <summary>The width of a cache entry (dimension 3), or 1 when the model leaves it symbolic.</summary>
            public int HeadDim { get; }

            /// <summary>Stores a validated layout; see <see cref="Read"/>.</summary>
            /// <param name="pastNames">The cache inputs.</param>
            /// <param name="presentNames">The matching cache outputs.</param>
            /// <param name="logitsName">The logits output.</param>
            /// <param name="heads">The heads of a cache entry.</param>
            /// <param name="headDim">The width of a cache entry.</param>
            private DecoderLayout(string[] pastNames, string[] presentNames, string logitsName, int heads, int headDim)
            {
                PastNames = pastNames;
                PresentNames = presentNames;
                LogitsName = logitsName;
                OutputNames = presentNames.Prepend(logitsName).ToArray();
                Heads = heads;
                HeadDim = headDim;
            }

            /// <summary>Reads and checks the layout of <paramref name="session"/>'s model.</summary>
            /// <param name="session">The decoder session.</param>
            /// <returns>The layout.</returns>
            /// <exception cref="NotSupportedException">The model is not an Optimum merged decoder (an input or output this
            /// search feeds or fetches is missing, or a cache input is not rank 4); the message names what is missing.</exception>
            public static DecoderLayout Read(InferenceSession session)
            {
                foreach (string required in new[] { "input_ids", "encoder_hidden_states", "use_cache_branch" })
                {
                    if (!session.InputMetadata.ContainsKey(required))
                        throw new NotSupportedException($"The decoder model has no '{required}' input; an Optimum merged decoder (decoder_model_merged.onnx) is expected.");
                }

                string[] pastNames = session.InputNames.Where(name => name.StartsWith(PastPrefix, StringComparison.Ordinal)).ToArray();
                if (pastNames.Length == 0)
                    throw new NotSupportedException($"The decoder model has no '{PastPrefix}*' input; a decoder with a key/value cache is expected.");
                string[] presentNames = pastNames.Select(name => PresentPrefix + name.Substring(PastPrefix.Length)).ToArray();
                for (int i = 0; i < pastNames.Length; i++)
                {
                    if (!session.OutputMetadata.ContainsKey(presentNames[i]))
                        throw new NotSupportedException($"The decoder model has the cache input '{pastNames[i]}' but no '{presentNames[i]}' output to continue it.");
                }

                // (batch, heads, past length, head width): the first step's placeholder takes the fixed extents.
                int[] dims = session.InputMetadata[pastNames[0]].Dimensions;
                if (dims.Length != 4)
                    throw new NotSupportedException($"The cache input '{pastNames[0]}' has rank {dims.Length}; (batch, heads, length, width) is expected.");
                string logitsName = session.OutputMetadata.ContainsKey("logits") ? "logits" : session.OutputNames[0];
                return new DecoderLayout(pastNames, presentNames, logitsName, Math.Max(dims[1], 1), Math.Max(dims[3], 1));
            }
        }
    }
}
