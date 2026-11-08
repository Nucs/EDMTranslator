using System.Text.Json;
using System.Text.Json.Serialization;
using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Xunit.Abstractions;

namespace EDMTranslatorTest
{
    /// <summary>
    /// The transformers reference the parity tests compare against: <c>reference_translations.json</c>, written by
    /// <c>tools/reference/generate_reference.py</c> from <c>EncoderDecoderModel.generate(num_beams=1, do_sample=False)</c>
    /// on each model's PyTorch weights.
    /// </summary>
    public sealed class ReferenceTranslations
    {
        /// <summary>The reference per model key (<c>jesc</c>, <c>ffxiv</c>, <c>aihub</c>).</summary>
        [JsonPropertyName("models")]
        public Dictionary<string, ModelReference> Models { get; set; } = new();

        /// <summary>Reads <c>reference_translations.json</c> from the test binaries' folder.</summary>
        /// <returns>The reference.</returns>
        /// <exception cref="FileNotFoundException">The file was not copied next to the test binaries.</exception>
        /// <exception cref="JsonException">The file is not valid JSON of this shape.</exception>
        public static ReferenceTranslations Load() =>
            JsonSerializer.Deserialize<ReferenceTranslations>(File.ReadAllText("reference_translations.json"))
            ?? throw new JsonException("reference_translations.json is empty.");
    }

    /// <summary>One model's reference: the ids its search starts and stops at, and the sentences.</summary>
    public sealed class ModelReference
    {
        /// <summary>The model's Hugging Face id.</summary>
        [JsonPropertyName("hub")]
        public string Hub { get; set; } = "";

        /// <summary>The decoder start token the search feeds first.</summary>
        [JsonPropertyName("decoder_start_token_id")]
        public long DecoderStartTokenId { get; set; }

        /// <summary>The token that ends the search.</summary>
        [JsonPropertyName("eos_token_id")]
        public long EosTokenId { get; set; }

        /// <summary>The translator's step limit.</summary>
        [JsonPropertyName("max_new_tokens")]
        public int MaxNewTokens { get; set; }

        /// <summary>The sentences with their encoder input and the reference output.</summary>
        [JsonPropertyName("rows")]
        public List<ReferenceRow> Rows { get; set; } = new();
    }

    /// <summary>One sentence of the reference.</summary>
    public sealed class ReferenceRow
    {
        /// <summary>The Japanese source sentence.</summary>
        [JsonPropertyName("sentence")]
        public string Sentence { get; set; } = "";

        /// <summary>The encoder input ids the reference was computed from (what <see cref="BertJa2GPTTokenizer"/> produces).</summary>
        [JsonPropertyName("input_ids")]
        public int[] InputIds { get; set; } = Array.Empty<int>();

        /// <summary>The generated ids without the decoder start token, EOS included when reached.</summary>
        [JsonPropertyName("output_ids")]
        public uint[] OutputIds { get; set; } = Array.Empty<uint>();
    }

    /// <summary>
    /// A <see cref="BertJa2GPTTokenizer"/> that records the sequence each translation decodes. It re-implements
    /// <see cref="ITokenizer"/>, and the translators call <c>Tokenizer.Decode</c> through that interface, so the
    /// recording sees exactly the ids the greedy search generated.
    /// </summary>
    public sealed class RecordingTokenizer : BertJa2GPTTokenizer, ITokenizer
    {
        /// <summary>The ids of the last <c>Decode</c> call through <see cref="ITokenizer"/>.</summary>
        public uint[] LastDecoded { get; private set; } = Array.Empty<uint>();

        /// <summary>Builds the underlying tokenizer.</summary>
        /// <param name="encoderDictDir">The MeCab dictionary folder.</param>
        /// <param name="encoderVocabPath">The encoder's <c>vocab.txt</c>.</param>
        /// <param name="decoderVocabPath">The decoder's <c>tokenizer.json</c>.</param>
        public RecordingTokenizer(string encoderDictDir, string encoderVocabPath, string decoderVocabPath)
            : base(encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath, decoderVocabPath: decoderVocabPath) { }

        /// <summary>Encodes through the base tokenizer.</summary>
        /// <param name="sentence">The source sentence.</param>
        /// <returns>The input ids and the attention mask.</returns>
        (int[], int[]) ITokenizer.Encode(string sentence) => Encode(sentence);

        /// <summary>Records <paramref name="tokens"/>, then decodes them through the base tokenizer.</summary>
        /// <param name="tokens">The generated ids.</param>
        /// <returns>The decoded text.</returns>
        string ITokenizer.Decode(uint[] tokens)
        {
            LastDecoded = tokens;
            return Decode(tokens);
        }
    }

    /// <summary>The steps every parity test shares: build a recording tokenizer, translate the reference, compare.</summary>
    public static class ReferenceParity
    {
        /// <summary>Builds a <see cref="RecordingTokenizer"/> the way the other tests build their tokenizer (vocabularies from the hub).</summary>
        /// <param name="encoderDictDir">The MeCab dictionary folder.</param>
        /// <param name="decoderHubName">The hub model whose <c>tokenizer.json</c> decodes the output.</param>
        /// <returns>The tokenizer.</returns>
        /// <exception cref="HttpRequestException">A vocabulary could not be downloaded (and is not cached yet).</exception>
        public static async Task<RecordingTokenizer> CreateTokenizerAsync(string encoderDictDir, string decoderHubName)
        {
            var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
            var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(decoderHubName, "tokenizer.json", "deps");
            return new RecordingTokenizer(encoderDictDir, encoderVocabPath, decoderVocabPath);
        }

        /// <summary>
        /// Translates every reference sentence of <paramref name="modelKey"/> and asserts that the tokenizer still produces
        /// the reference's input ids and that the search generates exactly the reference's output ids.
        /// </summary>
        /// <param name="modelKey">The model's key in the reference file.</param>
        /// <param name="tokenizer">The recording tokenizer <paramref name="translator"/> was built over.</param>
        /// <param name="translator">The translator under test.</param>
        /// <param name="output">Where each translation is logged.</param>
        /// <exception cref="Xunit.Sdk.XunitException">An input or an output differs from the reference.</exception>
        public static void AssertMatchesReference(string modelKey, RecordingTokenizer tokenizer, ITranslator translator, ITestOutputHelper output)
        {
            ModelReference reference = ReferenceTranslations.Load().Models[modelKey];
            foreach (ReferenceRow row in reference.Rows)
            {
                // The reference was computed from these ids; a tokenizer change would make the comparison meaningless.
                var (inputIds, _) = tokenizer.Encode(row.Sentence);
                Assert.Equal(row.InputIds, inputIds);

                string translated = translator.Translate(row.Sentence);
                output.WriteLine($"{row.Sentence} -> {translated}");
                Assert.Equal(row.OutputIds, tokenizer.LastDecoded);
            }
        }
    }
}
