using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Xunit.Abstractions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    /// <summary>
    /// Integration tests of <see cref="FF14JaKoTranslator"/> on the real model (paths from <see cref="TestConfigFile"/>):
    /// the tokenizer round trip, sample translations, and token-for-token parity with transformers.
    /// </summary>
    [Collection("Sequential")]
    public class FF14JaKoTranslatorTest : IDisposable
    {
        private BertJa2GPTTokenizer? tokenizer;
        private FF14JaKoTranslator? translator;
        private readonly string encoderDictDir;
        private readonly string modelDir;

        private readonly ITestOutputHelper output;

        public FF14JaKoTranslatorTest(ITestOutputHelper output)
        {
            this.output = output;

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            // read test_config.yaml
            using (var reader = new StreamReader(TestConfigFile.Path))
            {
                var yaml = deserializer.Deserialize<Dictionary<string, FF14JaKoTranslatorConfig>>(reader);
                var config = yaml["ff14_ja_ko_translator"];

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
            var recorder = await ReferenceParity.CreateTokenizerAsync(encoderDictDir, "skt/kogpt2-base-v2");
            using var parityTranslator = new FF14JaKoTranslator(recorder, modelDir);
            ReferenceParity.AssertMatchesReference("ffxiv", recorder, parityTranslator, output);
        }

        private async Task<BertJa2GPTTokenizer> InitTokenizerAsync()
        {
            // Prepare the tokenizer
            var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
            var hubName = "skt/kogpt2-base-v2";
            var decoderVocabFilename = "tokenizer.json";
            var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

            return new BertJa2GPTTokenizer(
                               encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
                                              decoderVocabPath: decoderVocabPath);
        }

        private FF14JaKoTranslator InitTranslator()
        {
            // Prepare the translator
            return new FF14JaKoTranslator(tokenizer, modelDir);
        }

        private void TestTokenizer(ITokenizer tokenizer)
        {
            output.WriteLine("--Tokenizer test--");
            output.WriteLine("[Encode]");
            var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
            output.WriteLine($"Input: {sentenceJa}");
            var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
            output.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

            output.WriteLine("[Decode]");
            // Tokens of "음, 이제 식사도 해볼까요"
            var tokens = new uint[] { 9330, 387, 12857, 9376, 18649, 9098, 7656, 6969, 8084, 1 };
            output.WriteLine($"Input: {string.Join(", ", tokens)}");
            var decoded = tokenizer.Decode(tokens);
            output.WriteLine($"Decoded: {decoded}");
        }

        private void TestTranslator(FF14JaKoTranslator translator)
        {
            output.WriteLine("--Translator test--");
            Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
            Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
            Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
            Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
        }

        private void Translate(FF14JaKoTranslator translator, string sentence)
        {
            output.WriteLine($"SourceText: {sentence}");
            string translated = translator.Translate(sentence);
            output.WriteLine($"Translated: {translated}");
        }

        /// <summary>
        /// Releases the translator's sessions after each test: the GC does not see their native memory (about 1.5 GB), so it
        /// would not hurry to.
        /// </summary>
        public void Dispose()
        {
            translator?.Dispose();
            tokenizer = null;
            translator = null;
        }
    }
}