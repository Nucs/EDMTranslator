using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using NumSharp;

namespace EDMTranslator.Models
{
    public abstract class EncoderDecoderTranslator : Translator
    {
        public EncoderDecoderTranslator(string modelName, string modelHubName, ITokenizer tokenizer) :
            base(modelInfo: new ModelInfo(
                modelName: modelName,
                modelHubName: modelHubName,
                modelType: ModelType.EncoderDecoder,
                modelUri: $"https://huggingface.co/{modelHubName}"),
            tokenizer)
        {

        }

        protected static Tensor<float> InitKeyValues(int inputIdLength)
        {
            // NumSharp 0.80 keeps NumPy's one signature, np.zeros(shape, dtype); the generic params-int[] overload is gone.
            return np.zeros(new Shape(1, 12, inputIdLength, 64), np.float32).ToMuliDimArray<float>().ToTensor<float>();
        }

        protected string GreedySearch(List<NamedOnnxValue> decoderInput, InferenceSession decoderSession,
            long bosTokenId, long eosTokenId, int maxLength = 50)
        {
            // Initialize the input for the decoder with the BOS token ID
            var inputIds = new long[] { bosTokenId };
            var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
            // Update input_ids for the decoder
            decoderInput[0] = NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor);

            // Initialize the list to store the generated tokens
            List<long> generatedTokens = new();
            IEnumerable<IEnumerable<DisposableNamedOnnxValue>> packedPastKeyValues = new List<List<DisposableNamedOnnxValue>>();

            // Greedy search loop
            for (int i = 0; i < maxLength; i++)
            {
                // Run the decoder model
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults;
                if (i == 0)
                {
                    (decoderResults, packedPastKeyValues) = ForwardOnnx(decoderSession, decoderInput);
                }
                else
                {
                    (decoderResults, packedPastKeyValues) = ForwardOnnx(decoderSession, decoderInput, packedPastKeyValues);
                }

                // Get the logits from the decoder results
                var logits = decoderResults.First().AsTensor<float>();

                // Apply softmax to logits to get probabilities
                var probabilities = Softmax(logits);
                var nextTokenId = ArgMax(probabilities.ToTensor());

                // Append the token to the list
                generatedTokens.Add(nextTokenId);

                // Prepare the input for the next iteration
                inputIdsTensor = new DenseTensor<long>(new long[] { nextTokenId }, new[] { 1, 1 });
                decoderInput.Find(input => input.Name.Equals("input_ids")).Value = inputIdsTensor;

                //decoderResults.Dispose();
                // Check if EOS token is generated
                if (nextTokenId == eosTokenId)
                {
                    break;
                }
            }
            // Dispose the packedPastKeyValues
            foreach (var item in packedPastKeyValues)
            {
                foreach (var value in item)
                {
                    value.Dispose();
                }
            }

            // Decode the generated tokens into text
            var generatedText = Tokenizer.Decode(generatedTokens.Select(item => (uint)item).ToArray());
            return generatedText;
        }

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

        protected static float[] Softmax(Tensor<float> logits)
        {
            // Find the maximum value in the logits
            float maxLogit = logits.Max();

            // Subtract the max value from each logit to avoid overflow during the exponential operation
            float[] expLogits = logits.Select(logit => (float)Math.Exp(logit - maxLogit)).ToArray();

            // Calculate the sum of the exponentials
            float sumExpLogits = expLogits.Sum();

            // Divide each exponential by the sum to get the softmax probabilities
            float[] softmaxProbabilities = expLogits.Select(expLogit => expLogit / sumExpLogits).ToArray();

            return softmaxProbabilities;
        }

        protected static int ArgMax(Tensor<float> probabilities, bool getSecond = false)
        {
            int argMaxIndex = 0;
            if (getSecond)
            {
                // Find the index of the second maximum value in the probabilities
                argMaxIndex = Array.IndexOf(probabilities.ToArray(), probabilities.OrderByDescending(x => x).Skip(1).First());
                return argMaxIndex;
            }
            // Find the index of the maximum value in the probabilities
            argMaxIndex = Array.IndexOf(probabilities.ToArray(), probabilities.Max());

            return argMaxIndex;
        }
    }
}
