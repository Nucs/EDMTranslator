using EDMTranslator.Models;
using EDMTranslator.Tokenization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using Xunit.Abstractions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    /// <summary>
    /// The protected members <see cref="EncoderDecoderTranslator"/> had before the NumSharp 0.80 search, exercised the way
    /// a subclass written against EDMTranslator 1.3.1 uses them: the <see cref="NamedOnnxValue"/> greedy search, and the
    /// 1.3.1 search loop rebuilt on <c>ForwardOnnx</c> / <c>Softmax</c> / <c>ArgMax</c>. Both must still translate exactly
    /// like transformers. The helper tests at the bottom need no model.
    /// </summary>
    [Collection("Sequential")]
    public class LegacySurfaceTest
    {
        private readonly ITestOutputHelper output;

        /// <summary>Takes xunit's output sink.</summary>
        /// <param name="output">Where the translations are logged.</param>
        public LegacySurfaceTest(ITestOutputHelper output)
        {
            this.output = output;
        }

        /// <summary>
        /// A 1.3.1-style subclass that builds its decoder inputs as <see cref="NamedOnnxValue"/>s and calls
        /// <c>GreedySearch(List&lt;NamedOnnxValue&gt;, ...)</c> translates like transformers.
        /// </summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task NamedOnnxValueGreedySearchMatchesReferenceAsync()
        {
            var (recorder, translator) = await CreateAsync();
            using (translator)
                ReferenceParity.AssertMatchesReference("jesc", recorder, translator, output);
        }

        /// <summary>
        /// The 1.3.1 greedy loop itself, run on the kept helpers (<c>ForwardOnnx</c> for both branches, <c>Softmax</c>,
        /// <c>ArgMax</c>), translates like transformers: the cache fix lives in <c>ForwardOnnx</c>, not only in the new search.
        /// </summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task OneThreeOneLoopOnTheKeptHelpersMatchesReferenceAsync()
        {
            var (recorder, translator) = await CreateAsync();
            using (translator)
            {
                translator.UseOneThreeOneLoop = true;
                ReferenceParity.AssertMatchesReference("jesc", recorder, translator, output);
            }
        }

        /// <summary>The merged GPT-2 decoder takes two cache tensors per layer (key and value).</summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task PastValuesPerLayerReadsTwoForTheGpt2DecoderAsync()
        {
            var (_, translator) = await CreateAsync();
            using (translator)
                Assert.Equal(2, LegacySurfaceTranslator.PastValuesPerLayerOf(translator.DecoderSession));
        }

        /// <summary><c>Softmax</c> equals the 1.3.1 LINQ softmax within float32 rounding and sums to 1.</summary>
        [Fact]
        public void SoftmaxMatchesTheLinqVersion()
        {
            float[] logits = RandomLogits(50257, seed: 7);
            var tensor = new DenseTensor<float>(logits, new[] { 1, 1, logits.Length });
            float[] got = LegacySurfaceTranslator.SoftmaxOf(tensor);
            float[] want = LinqSoftmax(tensor);
            Assert.Equal(want.Length, got.Length);
            for (int i = 0; i < want.Length; i++)
                Assert.InRange(Math.Abs(got[i] - want[i]), 0f, 1e-5f * want[i] + 1e-12f);
            Assert.InRange(got.Sum(), 0.9999f, 1.0001f);
        }

        /// <summary><c>ArgMax</c> equals the 1.3.1 LINQ argmax: the first maximum, and the runner-up with <c>getSecond</c>.</summary>
        [Fact]
        public void ArgMaxMatchesTheLinqVersion()
        {
            float[] values = RandomLogits(51200, seed: 11);
            var tensor = new DenseTensor<float>(values, new[] { 1, 1, values.Length });
            Assert.Equal(LinqArgMax(values, getSecond: false), LegacySurfaceTranslator.ArgMaxOf(tensor, getSecond: false));
            Assert.Equal(LinqArgMax(values, getSecond: true), LegacySurfaceTranslator.ArgMaxOf(tensor, getSecond: true));

            // A repeated maximum: the first occurrence wins, and the "second" value is the maximum again.
            var tied = new float[] { 1f, 5f, 3f, 5f, 2f };
            var tiedTensor = new DenseTensor<float>(tied, new[] { tied.Length });
            Assert.Equal(1, LegacySurfaceTranslator.ArgMaxOf(tiedTensor, getSecond: false));
            Assert.Equal(LinqArgMax(tied, getSecond: true), LegacySurfaceTranslator.ArgMaxOf(tiedTensor, getSecond: true));
        }

        /// <summary><c>InitKeyValues</c> is a zero tensor of the 12-head, 64-wide cache shape.</summary>
        [Fact]
        public void InitKeyValuesIsAZeroCacheEntry()
        {
            Tensor<float> kv = LegacySurfaceTranslator.KeyValues(3);
            Assert.Equal(new[] { 1, 12, 3, 64 }, kv.Dimensions.ToArray());
            Assert.All(kv, value => Assert.Equal(0f, value));
        }

        /// <summary>Builds the JESC tokenizer (recording) and the 1.3.1-style translator from the test config.</summary>
        /// <returns>The tokenizer and the translator over it.</returns>
        private static async Task<(RecordingTokenizer, LegacySurfaceTranslator)> CreateAsync()
        {
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            JESCJaEnTranslatorConfig config;
            using (var reader = new StreamReader(TestConfigFile.Path))
                config = deserializer.Deserialize<Dictionary<string, JESCJaEnTranslatorConfig>>(reader)["jesc_ja_en_translator"];
            var recorder = await ReferenceParity.CreateTokenizerAsync(config.EncoderDictDir, "openai-community/gpt2");
            return (recorder, new LegacySurfaceTranslator(recorder, config.ModelDir));
        }

        /// <summary>Deterministic logits in [-10, 10).</summary>
        /// <param name="length">How many.</param>
        /// <param name="seed">The generator seed.</param>
        /// <returns>The logits.</returns>
        private static float[] RandomLogits(int length, int seed)
        {
            var rng = new Random(seed);
            var values = new float[length];
            for (int i = 0; i < length; i++)
                values[i] = (float)(rng.NextDouble() * 20 - 10);
            return values;
        }

        /// <summary>EDMTranslator 1.3.1's <c>Softmax</c>, verbatim.</summary>
        /// <param name="logits">The logits.</param>
        /// <returns>The probabilities.</returns>
        private static float[] LinqSoftmax(Tensor<float> logits)
        {
            float maxLogit = logits.Max();
            float[] expLogits = logits.Select(logit => (float)Math.Exp(logit - maxLogit)).ToArray();
            float sumExpLogits = expLogits.Sum();
            return expLogits.Select(expLogit => expLogit / sumExpLogits).ToArray();
        }

        /// <summary>EDMTranslator 1.3.1's <c>ArgMax</c>, verbatim.</summary>
        /// <param name="values">The scores.</param>
        /// <param name="getSecond">The runner-up instead of the maximum.</param>
        /// <returns>The index.</returns>
        private static int LinqArgMax(float[] values, bool getSecond)
        {
            if (getSecond)
                return Array.IndexOf(values, values.OrderByDescending(x => x).Skip(1).First());
            return Array.IndexOf(values, values.Max());
        }
    }

    /// <summary>
    /// A JESC translator written the way a subclass of EDMTranslator 1.3.1 was: <see cref="NamedOnnxValue"/> inputs, plain
    /// <see cref="InferenceSession"/>s, and the protected helpers. It also exposes the protected static helpers to the tests.
    /// </summary>
    internal sealed class LegacySurfaceTranslator : EncoderDecoderTranslator
    {
        private readonly InferenceSession encoderSession;

        /// <summary>The decoder session, for <see cref="PastValuesPerLayerOf"/>.</summary>
        public InferenceSession DecoderSession { get; }

        /// <summary><c>true</c> runs the 1.3.1 loop on the helpers; <c>false</c> calls <c>GreedySearch(List&lt;NamedOnnxValue&gt;, ...)</c>.</summary>
        public bool UseOneThreeOneLoop { get; set; }

        /// <summary>Opens the JESC model.</summary>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="modelDirPath">The folder holding the two ONNX files.</param>
        public LegacySurfaceTranslator(BertJa2GPTTokenizer tokenizer, string modelDirPath)
            : base("jesc-ja-en-translator", "sappho192/jesc-ja-en-translator", tokenizer)
        {
            using var sessionOptions = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
            encoderSession = new InferenceSession(Path.Combine(modelDirPath, "encoder_model.onnx"), sessionOptions);
            DecoderSession = new InferenceSession(Path.Combine(modelDirPath, "decoder_model_merged.onnx"), sessionOptions);
        }

        /// <summary>EDMTranslator 1.3.1's <c>JESCJaEnTranslator.Translate</c> (NumSharp 0.80's <c>np.zeros</c> aside), then the chosen search.</summary>
        /// <param name="sentence">The Japanese text.</param>
        /// <returns>The English translation.</returns>
        public override string Translate(string sentence)
        {
            (var inputIds, var attentionMask) = Tokenizer.Encode(sentence);

            NDArray ndInputIds = np.array(inputIds);
            ndInputIds = np.expand_dims(ndInputIds, 0);
            NDArray ndAttentionMask = np.array(attentionMask.ToArray());
            ndAttentionMask = np.expand_dims(ndAttentionMask, 0);

            var encoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("attention_mask", ndAttentionMask.ToMuliDimArray<long>().ToTensor<long>())
            };

            using var encoderResults = encoderSession.Run(encoderInput);
            var encoderResult = encoderResults[0];

            var useCacheBranch = np.array(new bool[] { false });
            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderResult.AsTensor<float>()),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch.ToMuliDimArray<bool>().ToTensor<bool>())
            };

            return UseOneThreeOneLoop
                ? OneThreeOneGreedySearch(decoderInput, 50256, 50256, 1024)
                : GreedySearch(decoderInput, DecoderSession, 50256, 50256, 1024);
        }

        /// <summary>
        /// EDMTranslator 1.3.1's greedy loop on <c>ForwardOnnx</c> / <c>Softmax</c> / <c>ArgMax</c>, except that it disposes
        /// every step's results at the end (1.3.1 left them to the finalizer).
        /// </summary>
        /// <param name="decoderInput">The decoder inputs; modified as 1.3.1 modified them.</param>
        /// <param name="bosTokenId">The decoder start token.</param>
        /// <param name="eosTokenId">The token that ends the search.</param>
        /// <param name="maxLength">The most tokens to generate.</param>
        /// <returns>The decoded translation.</returns>
        private string OneThreeOneGreedySearch(List<NamedOnnxValue> decoderInput, long bosTokenId, long eosTokenId, int maxLength)
        {
            var inputIds = new long[] { bosTokenId };
            var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
            decoderInput[0] = NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor);

            List<long> generatedTokens = new();
            IEnumerable<IEnumerable<DisposableNamedOnnxValue>> packedPastKeyValues = new List<List<DisposableNamedOnnxValue>>();
            // Later steps read earlier steps' outputs as their cache, so every result stays alive until the search ends.
            var allResults = new List<IDisposable>();
            try
            {
                for (int i = 0; i < maxLength; i++)
                {
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults;
                    if (i == 0)
                        (decoderResults, packedPastKeyValues) = ForwardOnnx(DecoderSession, decoderInput);
                    else
                        (decoderResults, packedPastKeyValues) = ForwardOnnx(DecoderSession, decoderInput, packedPastKeyValues);
                    allResults.Add(decoderResults);

                    var logits = decoderResults.First().AsTensor<float>();
                    var probabilities = Softmax(logits);
                    var nextTokenId = ArgMax(probabilities.ToTensor());
                    generatedTokens.Add(nextTokenId);

                    inputIdsTensor = new DenseTensor<long>(new long[] { nextTokenId }, new[] { 1, 1 });
                    decoderInput.Find(input => input.Name.Equals("input_ids"))!.Value = inputIdsTensor;
                    if (nextTokenId == eosTokenId)
                        break;
                }
            }
            finally
            {
                foreach (IDisposable results in allResults)
                    results.Dispose();
            }

            return Tokenizer.Decode(generatedTokens.Select(item => (uint)item).ToArray());
        }

        /// <summary>The protected <c>Softmax</c>.</summary>
        /// <param name="logits">The logits.</param>
        /// <returns>The probabilities.</returns>
        public static float[] SoftmaxOf(Tensor<float> logits) => Softmax(logits);

        /// <summary>The protected <c>ArgMax</c>.</summary>
        /// <param name="probabilities">The scores.</param>
        /// <param name="getSecond">The runner-up instead of the maximum.</param>
        /// <returns>The index.</returns>
        public static int ArgMaxOf(Tensor<float> probabilities, bool getSecond) => ArgMax(probabilities, getSecond);

        /// <summary>The protected <c>InitKeyValues</c>.</summary>
        /// <param name="length">The placeholder's sequence length.</param>
        /// <returns>The zero tensor.</returns>
        public static Tensor<float> KeyValues(int length) => InitKeyValues(length);

        /// <summary>The protected <c>PastValuesPerLayer</c>.</summary>
        /// <param name="session">The decoder session.</param>
        /// <returns>Cache tensors per layer.</returns>
        public static int PastValuesPerLayerOf(InferenceSession session) => PastValuesPerLayer(session);

        /// <summary>Releases both sessions, the way a 1.3.1-style subclass adopts <see cref="EncoderDecoderTranslator.Dispose()"/>.</summary>
        /// <param name="disposing"><c>true</c> from <see cref="EncoderDecoderTranslator.Dispose()"/>.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                encoderSession.Dispose();
                DecoderSession.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
