using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Xunit.Abstractions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    /// <summary>
    /// Integration tests of <see cref="JESCJaEnTranslator"/> on the real model (paths from <see cref="TestConfigFile"/>):
    /// the tokenizer round trip, sample translations, and token-for-token parity with transformers.
    /// </summary>
    [Collection("Sequential")]
    public class JESCJaEnTranslatorTest : IDisposable
    {
        private BertJa2GPTTokenizer? tokenizer;
        private JESCJaEnTranslator? translator;
        private readonly string encoderDictDir;
        private readonly string modelDir;

        private readonly ITestOutputHelper output;

        public JESCJaEnTranslatorTest(ITestOutputHelper output)
        {
            this.output = output;

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            // read test_config.yaml
            using (var reader = new StreamReader(TestConfigFile.Path))
            {
                var yaml = deserializer.Deserialize<Dictionary<string, JESCJaEnTranslatorConfig>>(reader);
                var config = yaml["jesc_ja_en_translator"];

                encoderDictDir = config.EncoderDictDir;
                modelDir = config.ModelDir;
            }
        }

        [Fact]
        public async Task TestTranslatorAsync()
        {
            tokenizer ??= await InitTokenizerAsync();
            TestTokenizer(tokenizer);

            translator ??= InitTranslator();
            TestTranslator(translator);
        }

        /// <summary>
        /// Every sentence of <c>reference_translations.json</c> generates exactly the tokens transformers' own greedy
        /// <c>generate()</c> generates on the model's PyTorch weights.
        /// </summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task TranslationsMatchTransformersReferenceAsync()
        {
            var recorder = await ReferenceParity.CreateTokenizerAsync(encoderDictDir, "openai-community/gpt2");
            using var parityTranslator = new JESCJaEnTranslator(recorder, modelDir);
            ReferenceParity.AssertMatchesReference("jesc", recorder, parityTranslator, output);
        }

        /// <summary>
        /// The <c>SessionOptions</c> overload changes how ONNX Runtime runs, not what it computes: with 2 intra-op threads the
        /// translations still equal transformers' token for token, and the caller's options may be disposed right away.
        /// </summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task SessionOptionsOverloadMatchesTransformersReferenceAsync()
        {
            var recorder = await ReferenceParity.CreateTokenizerAsync(encoderDictDir, "openai-community/gpt2");
            JESCJaEnTranslator tuned;
            using (var options = new Microsoft.ML.OnnxRuntime.SessionOptions { IntraOpNumThreads = 2 })
                tuned = new JESCJaEnTranslator(recorder, modelDir, options);
            using (tuned)
                ReferenceParity.AssertMatchesReference("jesc", recorder, tuned, output);
        }

        /// <summary>
        /// A disposed translator refuses to translate with <see cref="ObjectDisposedException"/> instead of calling into a
        /// released native session, and a second <c>Dispose</c> does nothing.
        /// </summary>
        /// <returns>The test's task.</returns>
        [Fact]
        public async Task TranslateAfterDisposeThrowsObjectDisposedAsync()
        {
            tokenizer ??= await InitTokenizerAsync();
            var disposed = new JESCJaEnTranslator(tokenizer, modelDir);
            Assert.False(string.IsNullOrEmpty(disposed.Translate("逃げろ!")));
            disposed.Dispose();
            disposed.Dispose();
            Assert.Throws<ObjectDisposedException>(() => disposed.Translate("逃げろ!"));
        }

        private async Task<BertJa2GPTTokenizer> InitTokenizerAsync()
        {
            // Prepare the tokenizer
            var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
            var hubName = "openai-community/gpt2";
            var decoderVocabFilename = "tokenizer.json";
            var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

            return new BertJa2GPTTokenizer(
                               encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
                                              decoderVocabPath: decoderVocabPath);
        }

        private JESCJaEnTranslator InitTranslator()
        {
            return new JESCJaEnTranslator(tokenizer, modelDir);
        }

        private void TestTokenizer(BertJa2GPTTokenizer tokenizer)
        {
            output.WriteLine("--Tokenizer test--");
            output.WriteLine("[Encode]");
            var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
            output.WriteLine($"Input: {sentenceJa}");
            var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
            output.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

            output.WriteLine("[Decode]");
            // Tokens of "i was nervous before the exam, and i had a fever."
            var tokens = new uint[] { 72, 373, 10927, 878, 262, 2814, 11, 290, 1312, 550, 257, 17372, 13 };
            output.WriteLine($"Input: {string.Join(", ", tokens)}");
            var decoded = tokenizer.Decode(tokens);
            output.WriteLine($"Decoded: {decoded}");
        }

        private void TestTranslator(JESCJaEnTranslator translator)
        {
            output.WriteLine("--Translator test--");
            Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
            Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
            Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
            Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
        }

        private void Translate(JESCJaEnTranslator translator, string sentence)
        {
            output.WriteLine($"SourceText: {sentence}");
            string translated = translator.Translate(sentence);
            output.WriteLine($"Translated: {translated}");
        }

        /// <summary>
        /// Releases the translator's sessions after each test, as the other translator tests do: the GC does not see their
        /// native memory (about 1.5 GB), so it would not hurry to.
        /// </summary>
        public void Dispose()
        {
            translator?.Dispose();
            tokenizer = null;
            translator = null;
        }
    }
}
